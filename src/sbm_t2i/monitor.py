"""Training-time read-outs that do not touch the objective: a fixed-noise Inception FID of the EMA copies, and a step timer.

FIDMonitor (train.monitor, off unless `every` > 0). Every `every` global steps, for each EMA copy in `decays`:
- each rank draws `n_per_rank` one-step samples from its own fixed generator (seed + 7919 rank; rank 0 with the default
  seed 2066 = 42 + 2024 is exactly the b2_adm50k stream, so its read equals `python -m sbm_t2i.cli eval ... n=<n_per_rank>`), labels
  arange(n_per_rank) % 1000, omega 0.8, the b2_adm50k decode (sbm_t2i.eval.decoded_b2) and Inception-2048;
- the Inception moments are all-reduced, so the pooled read has world * n_per_rank samples (FID-8k on 4 GPUs) at the
  wall cost of one rank's share;
- rank 0 also scores its own n_per_rank samples, and the two points give FID_inf of FID(n) = FID_inf + k / n
  (arXiv 1911.07023).
Long-run read-outs, all on rank 0 from the pooled moments:
- `<tag>_fid50k_est` = FID_inf + k / 50000, the estimate of the protocol's FID-50k (the number the target is set in);
- `<tag>_mean` (the squared distance of the feature means: the grand-mean shift) and `<tag>_trace_ratio`
  (tr Sigma_gen / tr Sigma_real: below 1 the samples are under-dispersed);
- `label_scales`: each read repeated with the class rows of the label table moved away from the null row by that
  factor (tag suffix `_ls1_1`), the one-pass guidance of the read (sbm_t2i.eval label_scale);
- `steps_per_epoch` -> `monitor/epoch`; `schedule` = {metric, points [[step, value], ...]} -> `monitor/schedule`
  (piecewise linear in the step) and `monitor/gap_to_schedule` = metric - schedule (negative = ahead of the pace).
Inception is never in the loss, so this signal cannot be optimised against. Rank 0 appends one JSON line per read to
<run>/monitor/fid_monitor.jsonl and logs monitor/* to the run's logger. The copies are swapped in through the EMA
callback, sampled, and swapped back; no global RNG is consumed. Nothing is written to evals/ (these small-n numbers are
not protocol numbers).

TextMonitor (train.monitor with data.kind text_image). Every `every` steps, for each EMA copy in
`decays` and each label scale: rank r reads `n_per_rank` one-step samples T(2 z; prompt, omega) for captions
[r n, (r + 1) n) of MiniT2I's COCO-30K list from its fixed noise (seed + 7919 r); the Inception moments and the CLIP
cosines are all-reduced -> `<tag>_fid_coco` (against mscoco_fid_stats_512.npz; sbm_t2i.text_eval.Scorer: a trend, not
MiniT2I's protocol number), `<tag>_clip`, the mean term and trace ratio; rank 0 saves its first 16 samples as
monitor/samples_step<step>_<tag>.jpg. One JSON line per read in monitor/text_monitor.jsonl, monitor/* in the logger.

StepTimer (train.step_timer = N > 0): host wall seconds per optimizer step, averaged over N steps (the CUDA queue bounds
the error to well under a step), to monitor/ as perf.jsonl and perf/* in the logger.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


def fid_inf(n1, f1, n2, f2):
    """FID(n) = FID_inf + k / n through two reads (n1 < n2): (FID_inf, k)."""
    k = (f1 - f2) / (1.0 / n1 - 1.0 / n2)
    return f2 - k / n2, k


class FIDMonitor(Callback):
    def __init__(self, run_dir, every, n_per_rank=2000, decays=(0.999,), omega=0.8, seed=2066, batch=250, vae_batch=64,
                 refs=("val",), nets=None, ref_stats=None, label_scales=(1.0,), steps_per_epoch=None, schedule=None,
                 guidance_reads=()):
        """nets = (vae, feature_fn) and ref_stats = {name: (mu, sigma)} replace the real networks and statistics (tests)."""
        self.dir = Path(run_dir) / "monitor"
        self.every, self.n, self.decays = int(every), int(n_per_rank), tuple(float(d) for d in decays)
        if self.n % int(batch):
            raise ValueError(f"train.monitor.n_per_rank {self.n} must be a multiple of batch {batch}")
        self.p = dict(omega=float(omega), steps=[1], batch=int(batch), vae_batch=int(vae_batch), renoise_t=None,
                      guide_scale=0.0)
        self.seed, self.refs = int(seed), tuple(refs)
        self._nets, self._ref_stats, self._last = nets, ref_stats, -1
        self.label_scales = tuple(float(v) for v in (label_scales or (1.0,)))
        # for a map with a guidance input (model.gw_embedder): extra reads at [w, s_lo, s_hi] (tag suffix _w<w>)
        self.guidance_reads = tuple((float(w), float(lo), float(hi)) for w, lo, hi in (guidance_reads or ()))
        self.steps_per_epoch = None if not steps_per_epoch else float(steps_per_epoch)
        self.schedule = None
        if schedule and schedule.get("points"):
            pts = sorted((float(a), float(b)) for a, b in schedule["points"])
            self.schedule = (str(schedule["metric"]), pts)

    # ------------------------------------------------------------------ lazily built on the first read
    def _build(self, device):
        if self._nets is None:
            from torchmetrics.image.fid import FrechetInceptionDistance

            from sbm_t2i.eval import _ensure_inception_weights, _vae
            _ensure_inception_weights()
            inception = FrechetInceptionDistance(feature=2048, normalize=True).to(device).inception.eval()
            # FrechetInceptionDistance.update(normalize=True), op for op
            self._nets = (_vae(device), lambda imgs: inception((imgs * 255).byte()).double())
        if self._ref_stats is None:
            from sbm_t2i.eval import _ref
            self._ref_stats = {r: _ref(r, device) for r in self.refs}

    @staticmethod
    @contextmanager
    def _scaled_labels(model, label_scale):
        """The class rows of the label table moved away from the null (last) row by `label_scale`, restored on exit."""
        if float(label_scale) == 1.0:
            yield
            return
        w = model.y_embedder.embedding_table.weight.data
        keep = w.clone()
        w[:-1] = w[-1:] + float(label_scale) * (w[:-1] - w[-1:])
        try:
            yield
        finally:
            w.copy_(keep)

    @torch.no_grad()
    def moments(self, model, rank, device, scale, label_scale=1.0, guidance=None):
        """(sum, outer sum, count) of this rank's features, in float64."""
        from sbm_t2i.eval import decoded_b2
        vae, features = self._nets
        g = torch.Generator(device=device.type).manual_seed(self.seed + 7919 * int(rank))
        labels = torch.arange(self.n) % 1000
        p = self.p if guidance is None else {**self.p, "gw": guidance[0], "gw_lo": guidance[1], "gw_hi": guidance[2]}
        s = o = None
        with self._scaled_labels(model, label_scale):
            for _, images in decoded_b2(model, p, device, scale, vae, labels, g):
                f = features(images[1])
                f = f.unsqueeze(0) if f.dim() == 1 else f
                s = f.sum(dim=0) if s is None else s + f.sum(dim=0)
                o = f.t().mm(f) if o is None else o + f.t().mm(f)
        return s, o, torch.tensor(float(self.n), dtype=torch.float64, device=device)

    def _fid(self, s, o, n, ref):
        from sbm_t2i.eval import _fid, _mean_cov
        mu, cov = _mean_cov(s, o, n)
        return _fid(*self._ref_stats[ref], mu, cov)

    def _parts(self, s, o, n, ref):
        """(mean term, trace ratio) of the pooled moments against the reference."""
        from sbm_t2i.eval import _mean_cov
        mu, cov = _mean_cov(s, o, n)
        mu_r, cov_r = self._ref_stats[ref]
        return (float((mu.double() - mu_r.double()).pow(2).sum()),
                float(cov.double().diagonal().sum() / cov_r.double().diagonal().sum()))

    def _schedule_at(self, step):
        pts = self.schedule[1]
        if step <= pts[0][0]:
            return pts[0][1]
        for (a, va), (b, vb) in zip(pts[:-1], pts[1:]):
            if step <= b:
                return va + (vb - va) * (step - a) / max(b - a, 1e-9)
        return pts[-1][1]

    # ------------------------------------------------------------------ hooks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == self._last or step % self.every:
            return
        self._last = step
        self.read(trainer, pl_module, step)

    @torch.no_grad()
    def read(self, trainer, pl_module, step):
        from sbm_t2i.ema import EMA
        t0 = time.time()
        device = pl_module.device
        self._build(device)
        rank, world = int(trainer.global_rank), int(trainer.world_size)
        ema = next((c for c in trainer.callbacks if isinstance(c, EMA)), None)
        scale = float(pl_module.latents_scale.flatten()[0])
        was = pl_module.model.training
        pl_module.model.eval()
        row = {"step": step, "n_per_rank": self.n, "world": world, "seed": self.seed, "omega": self.p["omega"]}
        metrics = {}
        for decay in self.decays:
            base = f"ema{decay:g}".replace(".", "_")
            if ema is not None and decay in ema.decays:
                make_ctx = lambda d=decay: ema.average_parameters(pl_module, d)   # noqa: E731
            elif ema is None or decay == 0:
                make_ctx, base = nullcontext, "raw"
            else:
                print(f"FIDMonitor: EMA decay {decay} is not tracked; skipped")
                continue
            reads = [(ls, None) for ls in self.label_scales] + [(1.0, gr) for gr in self.guidance_reads]
            for ls, gr in reads:
                tag = base if ls == 1.0 else base + "_ls" + f"{ls:g}".replace(".", "_")
                if gr is not None:
                    tag = base + "_w" + f"{gr[0]:g}".replace(".", "_")
                with make_ctx():
                    s, o, n = self.moments(pl_module.model, rank, device, scale, ls, gr)
                local = (s.clone(), o.clone(), n.clone())
                if world > 1 and dist.is_available() and dist.is_initialized():
                    for x in (s, o, n):
                        dist.all_reduce(x)
                if rank == 0:
                    for ref in self.refs:
                        pooled = self._fid(s, o, float(n), ref)
                        mean, ratio = self._parts(s, o, float(n), ref)
                        row.update({f"{tag}_fid_{ref}_n{int(n)}": pooled, f"{tag}_mean_{ref}": mean,
                                    f"{tag}_trace_ratio_{ref}": ratio})
                        metrics.update({f"monitor/{tag}_fid_{ref}_pooled": pooled, f"monitor/{tag}_mean_{ref}": mean,
                                        f"monitor/{tag}_trace_ratio_{ref}": ratio})
                        if world > 1:
                            own = self._fid(*local[:2], float(local[2]), ref)
                            f_inf, k = fid_inf(float(local[2]), own, float(n), pooled)
                            est = f_inf + k / 50000.0
                            row.update({f"{tag}_fid_{ref}_n{int(local[2])}_rank0": own, f"{tag}_fid_inf_{ref}": f_inf,
                                        f"{tag}_k_{ref}": k, f"{tag}_fid50k_est_{ref}": est})
                            metrics.update({f"monitor/{tag}_fid_{ref}_rank0": own, f"monitor/{tag}_fid_inf_{ref}": f_inf,
                                            f"monitor/{tag}_fid50k_est_{ref}": est})
        pl_module.model.train(was)
        if rank == 0:
            if self.steps_per_epoch:
                row["epoch"] = metrics["monitor/epoch"] = step / self.steps_per_epoch
            if self.schedule is not None:
                target = self._schedule_at(step)
                row["schedule"] = metrics["monitor/schedule"] = target
                got = metrics.get(f"monitor/{self.schedule[0]}")
                if got is not None:
                    row["gap_to_schedule"] = metrics["monitor/gap_to_schedule"] = got - target
            row["seconds"] = time.time() - t0
            metrics["monitor/seconds"] = row["seconds"]
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "fid_monitor.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics, step=step)
            print(f"FIDMonitor step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        return row


class TextMonitor(Callback):
    def __init__(self, run_dir, every, n_per_rank=2500, decays=(0.999,), omega=0.95, seed=2066, batch=50, label_scales=(1.0,),
                 captions=None, ref_stats=None, clip=None, sampler="one_step", steps=32, cfg=2.0, **_):
        from sbm_t2i.text_eval import DEFAULTS
        self.dir = Path(run_dir) / "monitor"
        self.every, self.n, self.batch = int(every), int(n_per_rank), int(batch)
        # sampler euler (the loss.flow control): `steps` Euler steps with CFG `cfg` over the whole path (MiniT2I's sampler)
        self.sampler, self.steps, self.cfg = str(sampler), int(steps), float(cfg)
        if self.sampler not in ("one_step", "euler"):
            raise ValueError(f"train.monitor.sampler must be one_step or euler, got {self.sampler}")
        self.decays, self.omega, self.seed = tuple(float(d) for d in decays), float(omega), int(seed)
        self.label_scales = tuple(float(v) for v in (label_scales or (1.0,)))
        self.paths = {"captions": captions or DEFAULTS["captions"], "ref_stats": ref_stats or DEFAULTS["ref_stats"],
                      "clip": clip or DEFAULTS["clip"]}
        self._scorer = self._captions = self._tok = None
        self._last = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == self._last or step % self.every:
            return
        self._last = step
        self.read(trainer, pl_module, step)

    @torch.no_grad()
    def _sums(self, pl_module, rank, label_scale):
        """This rank's score sums and its first 16 samples, read from the module's current weights."""
        from sbm_t2i.text import tokenize
        from sbm_t2i.text_eval import NOISE_SCALE, euler_cfg, one_step, to_unit
        device, model, size = pl_module.device, pl_module.model, int(pl_module.cfg.model.input_size)
        g = torch.Generator(device=device.type).manual_seed(self.seed + 7919 * int(rank))
        prompts = self._captions[rank * self.n:(rank + 1) * self.n]
        self._scorer.reset()
        first = None
        for i in range(0, len(prompts), self.batch):
            chunk = prompts[i:i + self.batch]
            ids, mask = tokenize(self._tok, chunk, pl_module.cfg.data.get("text_len", 256))
            context, mask = pl_module._text_encoder(ids.to(device), mask.to(device))
            # a codec (sbm_t2i.codec): noise N(0, I) in its latent (the guided_cbm interpolant's), the read decoded to pixels
            codec = getattr(pl_module, "_codec", None)
            z = (1.0 if codec is not None else NOISE_SCALE) * torch.randn(len(chunk), model.in_channels, size, size,
                                                                         generator=g, device=device)
            if self.sampler == "euler":
                out = euler_cfg(model, context, mask, z, self.steps, self.cfg)
            else:
                out = one_step(model, context, mask, z, self.omega, pl_module.cfg.model.label_dim, label_scale)
            images = to_unit(out if codec is None else codec.decode(out))
            self._scorer.add(images, chunk)
            first = images[:16].clone() if first is None else first
        return [t.clone() for t in self._scorer.sums], first

    @torch.no_grad()
    def read(self, trainer, pl_module, step):
        from sbm_t2i.ema import EMA
        from sbm_t2i.text import tokenizer
        from sbm_t2i.text_eval import Scorer, load_captions, save_grid
        t0 = time.time()
        rank, world = int(trainer.global_rank), int(trainer.world_size)
        if self._scorer is None:
            self._scorer = Scorer(pl_module.device, self.paths["ref_stats"], self.paths["clip"])
            self._captions, self._tok = load_captions(self.paths["captions"]), tokenizer(pl_module.cfg.data.text_encoder)
            if len(self._captions) < world * self.n:
                raise ValueError(f"train.monitor: {world} ranks x n_per_rank {self.n} exceeds the {len(self._captions)} captions")
        ema = next((c for c in trainer.callbacks if isinstance(c, EMA)), None)
        was = pl_module.model.training
        pl_module.model.eval()
        row, metrics = {"step": step, "n_per_rank": self.n, "world": world, "seed": self.seed, "omega": self.omega}, {}
        if self.sampler == "euler":
            row.update(sampler="euler", steps=self.steps, cfg=self.cfg)
        for decay in self.decays:
            base = f"ema{decay:g}".replace(".", "_")
            if ema is not None and decay in ema.decays:
                make_ctx = lambda d=decay: ema.average_parameters(pl_module, d)   # noqa: E731
            elif ema is None or decay == 0:
                make_ctx, base = nullcontext, "raw"
            else:
                print(f"TextMonitor: EMA decay {decay} is not tracked; skipped")
                continue
            for ls in self.label_scales:
                tag = base if ls == 1.0 else base + "_ls" + f"{ls:g}".replace(".", "_")
                with make_ctx():
                    sums, first = self._sums(pl_module, rank, ls)
                if world > 1 and dist.is_available() and dist.is_initialized():
                    for x in sums:
                        dist.all_reduce(x)
                if rank == 0:
                    res = self._scorer.result(sums)
                    row.update({f"{tag}_{k}": v for k, v in res.items()})
                    metrics.update({f"monitor/{tag}_{k}": v for k, v in res.items() if k != "n"})
                    save_grid(first, self.dir / f"samples_step{step:07d}_{tag}.jpg")
        pl_module.model.train(was)
        if rank == 0:
            row["seconds"] = metrics["monitor/seconds"] = time.time() - t0
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "text_monitor.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics, step=step)
            print(f"TextMonitor step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        return row


class StepTimer(Callback):
    def __init__(self, run_dir, every, images_per_step):
        self.dir, self.every, self.images = Path(run_dir) / "monitor", int(every), int(images_per_step)
        self._t0 = self._s0 = None
        self._last = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == self._last:
            return
        self._last = step
        now = time.perf_counter()
        if self._t0 is None:
            self._t0, self._s0 = now, step
            return
        if step - self._s0 < self.every:
            return
        sec = (now - self._t0) / (step - self._s0)
        row = {"step": step, "steps": step - self._s0, "sec_per_step": sec,
               "img_per_sec_per_gpu": self.images / sec / max(int(trainer.world_size), 1)}
        self._t0, self._s0 = now, step
        if trainer.is_global_zero:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "perf.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if trainer.logger is not None:
                trainer.logger.log_metrics({"perf/sec_per_step": sec, "perf/img_per_sec_per_gpu": row["img_per_sec_per_gpu"]},
                                           step=step)
