"""The BTM / CBM objective of the class-conditional maps (loss.name cbm; the text-to-image recipes use loss.name guided_cbm,
sbm_t2i.guided_cbm). It is the objective of the SD-VAE ImageNet line (src/sbm/loss.py) with optional terms on top.

Notation: I_t = (1 - t) z + t x (t = 1 is data), T(.; y, omega) the map, omega the anchor time (network input).

    transport    || T(I_t) - sg(T(I_t) + J_T(I_t) (x - z)) ||^2 * sg[(d^2 + c)^-p]      (the JVP target, adaptive weight)
    endpoint     || T(I_omega') - x ||^2 * sg[(d^2 + c)^-p],   omega' ~ U[omega, 1]      (pins T = id on data)
                 endpoint.omega_skew k > 1 draws omega' = omega + (1 - omega)(1 - (1 - u)^k), closer to 1 (a sharper target);
                 endpoint.weight scales the endpoint's own gradient only (the balancer, CBM-E and Frechet keep the unweighted one).
    CBM-T        the transport term twice from the same z and t: once to x (weight 1 + alpha), once to a rolled x^- of
                 another class under the SAME label with the tangent negated (weight alpha). alpha = 0 is BTM.
                 `paired` is the null control: labels rolled with the targets, so the expected gradient is BTM's.
    CBM-E        at t <= t_max: push T(I_t, y) along T(I_t, y) - T(I_t, y^-), y^- rolled (CFG distilled into the map);
                 strength relative to the endpoint's full-parameter gradient norm.
    Frechet      (optional) the Frechet distance of DINOv2 features of fresh one-step samples to the train set's.
    aux          (optional) intermediate supervision: side readouts q_k after blocks model.aux_taps, projected
                 G_k = P_k q_k (P_k = avg-pool by aux.pool[k]): each gets the BTM rule in projected coordinates (the JVP
                 in the input space, targets sg(G_k + D_x G_k V), endpoint P_k x, the same weights and its own balancer
                 on its readout), summed as aux.alpha / K * sum_k; aux.align adds
                 mean_k MSE(G_k(I_t), P_k T(I_t)) with gradients through both. BTM only (signed alpha 0).
    guide        iMF-style guidance (Geng et al., improved MeanFlow) in the transport target: for transport
                 times t in [t_lo, t_hi] and non-null labels the tangent V = x - z becomes
                 w V + kappa b(I_t | y) + (1 - w - kappa) b(I_t | neg), with b(I_t | c) = (T(I_t; c, omega = t) - I_t) / (1 - t)
                 the map's own denoiser velocity (no gradient; neg = the null class, or a rolled class). In expectation the
                 CFG field b_{w + kappa}, so the fixed point transports along the guided flow. w = 1, kappa = g - 1 is the
                 shift form V + (g - 1)(b_y - b_neg), which keeps traction under the p = 1 weight where iMF's form
                 (kappa 0) is ~w^2 weaker. Needs the null class trained
                 (label_dropout > 0) and anchors down to t_lo (anchor_alpha >= 1 - t_lo). BTM only (signed alpha 0).

Optional switches (configs/t2i/train.yaml; every default keeps the base objective bit for bit):
    transport.p_within   within-t exponent p_w: the transport weight becomes sg[(d^2 + c)^-p_w (m_t + c)^-(p - p_w)],
                         m_t a running mean of d^2 in the example's logit-t bin (`bins` over logit t in [-4, 4], EMA
                         `bin_ema`, all-reduced over ranks). Inside a t slice the weight is (d^2 + c)^-p_w; across slices it
                         keeps p's scale. Positive branch only: the CBM-T negative branch keeps (d^2 + c)^-p.
    diag_every           every N steps, the full-network gradient norms of the weighted transport and endpoint terms,
                         kappa_full = |w_e g_E| / |w_t g_T| and their cosine (two extra backward passes on those steps).
    sync_free            the logged scalars are made on the device (no pageable host-to-device copy, which syncs).
    frechet.save_state   the Frechet term's running moments and fd_ref ride in the checkpoint, so a resume or a
                         continuation keeps the anneal instead of resetting fd_ref.
    gcond                guidance-strength-conditioned CBM-E (needs model.gw_embedder): each example draws w (1 + w ~ (1 + w)^-beta on [1 + w_min, 1 + w_max]) and
                         every term of it runs on the map at that w; CBM-E's force on it is w times the recipe's (form
                         force) or its target is iMF's guided target (form anchored); the Frechet batch of step s reads
                         at frechet_w[s mod K] with that slice's own running moments. w = 0 is BTM + Frechet.

Two evaluation paths share these definitions:
- `CBMLoss.__call__` (mode pretrain): one step's batch, draws from the global RNG in a fixed order, positive and negative
  branches in one concatenated JVP, balancer `match_smaller_to_larger` on the final layer inside the step (CBM-E and the
  Frechet term also apply on the signed rule).
- `pair_terms` (mode finetune): per-pair seeded draws, each branch its own forward, the unbalanced term means returned for
  the trainer to accumulate and balance.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sbm_t2i.frechet import FrechetTerm


def _bcast(t, like):
    return t.view(-1, *((1,) * (like.ndim - 1)))


def adaptive_mean(pred, target, p, c, sample_weight=None):
    """mean_b d_b^2 * sg[(d_b^2 + c)^-p] (* sample_weight_b), d_b^2 summed over the example; p = 0 is plain L2.
    The product is formed in the prediction's dtype (bf16 under autocast) and averaged in fp32."""
    delta_sq = ((pred - target) ** 2).sum(dim=tuple(range(1, pred.ndim)))
    weight = (1.0 / (delta_sq + c) ** p).detach()
    if sample_weight is not None:
        weight = weight * sample_weight.detach().reshape(-1)
    return torch.mean((delta_sq * weight).float())


def pair_means(pred, target, p, c):
    """Per-PAIR means of d^2 * sg[(d^2 + c)^-p], in the prediction's dtype (the reduction order matters in bf16)."""
    sq = (pred - target).square().flatten(1).sum(1)
    return (sq * (1 / (sq + c) ** p).detach()).reshape(-1, 2).mean(1)


def grad_norm(loss, parameters):
    """|| d loss / d parameters ||_2, the graph kept."""
    out = torch.zeros((), device=loss.device, dtype=torch.float32)
    if not loss.requires_grad or not parameters:
        return out
    for g in torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True):
        if g is not None:
            out += g.detach().float().pow(2).sum()
    return torch.sqrt(out)


class CBMLoss:
    def __init__(self, cfg, label_dim=1000, sync_free=False):
        """cfg: the `loss` block of the run config (configs/t2i/train.yaml)."""
        self.label_dim = int(label_dim)
        self.sync_free = bool(sync_free)
        self.tp, self.tc = float(cfg.transport.p), float(cfg.transport.c)
        pw = cfg.transport.get("p_within")
        self.p_within = None if pw is None else float(pw)
        self.t_bins, self.bin_ema = int(cfg.transport.get("bins", 16)), float(cfg.transport.get("bin_ema", 0.99))
        self._bin_mean = self._bin_seen = None
        self.diag_every = int(cfg.get("diag_every", 0) or 0)
        self.ep, self.ec = float(cfg.endpoint.p), float(cfg.endpoint.c)
        self.endpoint_weight = float(cfg.endpoint.get("weight", 1.0))
        self.omega_skew = float(cfg.endpoint.get("omega_skew", 1.0))
        # late_weight: endpoint rows at omega' > t0 weighted min{((1 - t0) / (1 - omega'))^power, cap}
        # (the epsilon-parametrisation's weight on the late band, where plain L2 on x leaves a residual-noise gradient ~ (1 - omega'))
        lw = cfg.endpoint.get("late_weight")
        self.late_weight = None if not lw else (float(lw["t0"]), float(lw.get("power", 2.0)), float(lw.get("cap", 64.0)))
        self.anchor_alpha = float(cfg.anchor_alpha)
        self.restrict = bool(cfg.restrict_transport)
        t = cfg.time
        self.p_mean0, self.p_std, self.eps = float(t.p_mean), float(t.p_std), float(t.eps)
        self.p_mean_end = None if t.get("p_mean_end") is None else float(t.p_mean_end)
        self.anneal_start, self.anneal_steps = int(t.get("anneal_start", 0)), int(t.get("anneal_steps", 0))
        self.p_mean = self.p_mean0
        # time.mix_frac: a fraction of the transport times uniform on [mix_lo, omega), the
        # late path the one-step read flows through (logit-normal -1.2 puts 5% of its mass on t > 0.6); t stays below
        # the example's anchor, so no transport sample lands in the endpoint band
        self.t_mix, self.t_mix_lo = float(t.get("mix_frac", 0.0)), float(t.get("mix_lo", 0.3))
        self.alpha, self.paired = float(cfg.signed.alpha), bool(cfg.signed.paired)
        if self.paired and self.alpha <= 0:
            raise ValueError("loss.signed.paired is the null control of the signed rule: it needs alpha > 0")
        self.balance = str(cfg.balance.mode)
        self.balance_path = str(cfg.balance.get("parameter_path", "final_layer"))
        # mode ema: the two final-layer norms are smoothed with this decay before the ratio is taken, so the transport
        # weight (~2000 under match_smaller_to_larger) no longer jitters x6 from step to step with the per-batch norms
        self.balance_ema = float(cfg.balance.get("ema", 0.99))
        self._balance_norms = None
        self.cfg_distill = bool(cfg.cfg_distill.enabled)
        self.cfg_strength, self.cfg_t_max = float(cfg.cfg_distill.strength), float(cfg.cfg_distill.t_max)
        self.label_dropout = float(cfg.get("label_dropout", 0.0))
        self.noise_scale = float(cfg.get("noise_scale", 1.0))   # z ~ noise_scale N(0, I) (MiniT2I's pixel interpolant: 2)
        # flow (a plain flow-matching baseline): no map, no transport, no anchors, no guide. The network is MiniT2I's
        # x-prediction denoiser D(I_t) trained with its v-loss |(D(I_t) - x) / max(1 - t, a_min)|^2 (mean over pixels),
        # t ~ logit-normal(time.p_mean, time.p_std), label dropout to the null prompt; read with Euler + CFG
        # (train.monitor.sampler euler).
        fl = cfg.get("flow") or {}
        self.flow, self.flow_a_min = bool(fl.get("enabled", False)), float(fl.get("a_min", 0.01))
        fr = cfg.frechet
        make_fr = lambda: FrechetTerm(fr.stats, vae_name=fr.vae, hub_dir=fr.dino_hub, ema=fr.ema, w_every=fr.w_every,
                                      fd_ref=fr.fd_ref)
        self.frechet = make_fr() if fr.enabled else None
        self.fr_strength, self.fr_n, self.fr_omega = float(fr.strength), int(fr.n_samples), float(fr.omega)
        self.fr_start = int(fr.get("start_step", 0))
        self.fr_save_state = bool(fr.get("save_state", False))
        # frechet.anneal false: no anneal (the push keeps its full share as FD falls); global_norm: the weight from the norm of
        # the rank-averaged Frechet gradient instead of this rank's few-sample one (whose noise inflates the norm)
        self.fr_anneal, self.fr_global_norm = bool(fr.get("anneal", True)), bool(fr.get("global_norm", False))
        self.transport_weight = float(cfg.transport.get("weight", 1.0))   # scales the weighted transport term only
        # fixed weights: when the endpoint is off or trivial its gradient norm is no usable reference scale
        fw, cw = fr.get("fixed_weight"), cfg.cfg_distill.get("fixed_weight")
        self.fr_fixed_weight = None if fw is None else float(fw)
        self.cfg_fixed_weight = None if cw is None else float(cw)
        ax = cfg.get("aux") or {}
        self.aux_alpha, self.aux_align = float(ax.get("alpha", 0.0)), float(ax.get("align", 0.0))
        self.aux_pool = tuple(int(k) for k in (ax.get("pool") or ()))
        self.aux_on = self.aux_alpha > 0 or self.aux_align > 0
        gd = cfg.get("guide") or {}
        self.guide = bool(gd.get("enabled", False))
        self.g_w, self.g_kappa = float(gd.get("w", 1.0)), float(gd.get("kappa", 0.0))
        self.g_tlo, self.g_thi = float(gd.get("t_lo", 0.35)), float(gd.get("t_hi", 0.6))
        self.g_negative, self.g_start = str(gd.get("negative", "uncond")), int(gd.get("start_step", 0))
        # iMF's guided boundary: inside the interval the endpoint target is
        # w x + kappa D(I_w' | y) + (1 - w - kappa) D(I_w' | neg), D the map's own diagonal read T(I_w'; c, omega = w');
        # diag_frac sets omega' = omega on that fraction of the endpoint rows (iMF's r = t rows)
        self.g_endpoint, self.g_diag_frac = bool(gd.get("endpoint", False)), float(gd.get("diag_frac", 0.0))
        # guide.diag_time: the diagonal rows' time omega' = omega ~ logit-normal(diag_p_mean,
        # diag_p_std) in [1 - anchor_alpha, 1) instead of the anchor's U[1 - anchor_alpha, 1]. They are the map's denoiser
        # (x-prediction at the interpolant's own time); at 512 px with noise 2 the prompt only matters at t <~ 0.3, which
        # U[0.02, 1] visits on ~29% of the diagonal rows and MiniT2I's flow law logit-normal(-0.8, 0.8) on ~48%. None = off
        dt = gd.get("diag_time")
        self.g_diag_time = None if not dt else (float(dt["p_mean"]), float(dt["p_std"]))
        # iMF's Omega-conditioning: per example w ~ w^-beta on [w_min, w_max] and an interval
        # s_lo ~ U[0, 0.5], s_hi ~ U[0.5, 1] (diagonal rows [0, 1]), fed to the map (model.gw_embedder with gw_input kappa,
        # model.gw_interval); inside the interval the tangent is V + (1 - 1/w)(b(y, w) - b(null, 1)) and the endpoint target
        # x + (1 - 1/w)(D(y, w) - D(null, 1)), the reads at omega = t (omega') with the interval [0, 1], no gradient
        self.g_omega_cond = bool(gd.get("omega_cond", False))
        self.g_wmin, self.g_wmax, self.g_beta = float(gd.get("w_min", 1.0)), float(gd.get("w_max", 8.0)), float(gd.get("beta", 1.0))
        # t_cap: the guided tangent only at t <= t_cap. Our velocity read is a denoiser displacement over (1 - t) (iMF reads a
        # velocity head), so near data the read's error is amplified by 1 / (1 - t); with s_hi up to 1 and no cap, training
        # can diverge in the guided phase. 1.0 = no cap. The guided endpoint has no division: not capped.
        self.g_tcap = float(gd.get("t_cap", 1.0))
        # guide.v_weight: the sample velocity x - z in the guided tangent is
        # replaced by the (teacher's) conditional velocity read with weight 1 - v_weight; 1 = the recipe, 0 = iMF's
        # predicted tangent b_y + kappa (b_y - b_neg); inside the guided interval only
        self.g_vw = float(gd.get("v_weight", 1.0))
        # guide.teacher_from: an external frozen teacher (e.g. a pretrained flow model, a time-blind denoiser) makes the
        # guide's reads: b_c - b_null is then the base's own CFG direction, the fixed point is its CFG flow at scale
        # 1 + kappa (not 1 / (1 - kappa): nothing is read back from the map), and the reads need no anchors at t
        self.g_external = bool(gd.get("teacher_from"))
        # guide.calib, evidence-calibrated guidance: w is set
        # online so the map's own class evidence on its one-step samples matches real data. Every step, n_gen kept rows:
        # real x and G = T(z; y, 0.8) are re-noised at a shared s ~ U[s_lo, s_hi] (shared noise) and read by the map at
        # y and null; evidence e = lambda(s)(|D_null - target|^2 - |D_y - target|^2), lambda(s) = s / (1 - s)^3. Every
        # `every` steps the all-reduced gap (real - gen; > 0: samples under-classed) feeds an EMA and
        # log w <- clip(log w + eta * every * gap_ema / sd_gap, log w_lo, log w_hi); then kappa = 1 - 1/w (w stays 1 in
        # the two-term form) and t_hi = min(t_hi, 1 - 0.6 kappa). The label gain of the bootstrapped rule is
        # gamma / (1 - gamma kappa), so a fixed kappa over-guides as the map learns. w is not checkpointed: on a relaunch,
        # set w_init to the logged guide_w.
        cal = gd.get("calib") or {}
        self.ec = self.guide and bool(cal.get("enabled", False))
        if self.ec:
            if not self.g_endpoint or self.g_negative != "uncond" or self.g_omega_cond or self.g_w != 1.0:
                raise ValueError("loss.guide.calib drives iMF's two-term rule: guide.w 1, endpoint true, negative uncond, "
                                 "no omega_cond")
            self.ec_wlo, self.ec_whi = float(cal.get("w_lo", 1.0)), float(cal.get("w_hi", 6.0))
            w0 = float(cal.get("w_init", 3.0))
            if not 1.0 <= self.ec_wlo <= w0 <= self.ec_whi:
                raise ValueError("loss.guide.calib needs 1 <= w_lo <= w_init <= w_hi")
            self.ec_logw = math.log(w0)
            self.ec_eta, self.ec_ema = float(cal.get("eta", 2e-4)), float(cal.get("ema", 0.99))
            self.ec_n, self.ec_every = int(cal.get("n_gen", 16)), max(1, int(cal.get("every", 10)))
            self.ec_slo, self.ec_shi = float(cal.get("s_lo", 0.35)), float(cal.get("s_hi", 0.6))
            self.ec_gap = self.ec_var = None
            self.ec_acc, self.ec_count, self.ec_last = None, 0, {}
            self.g_thi0 = self.g_thi
            self._ec_apply()
        if self.guide and self.g_omega_cond:
            if self.g_negative != "uncond" or not self.g_endpoint:
                raise ValueError("loss.guide.omega_cond is iMF's recipe: negative uncond and the guided endpoint (endpoint true)")
            if not 1.0 <= self.g_wmin <= self.g_wmax:
                raise ValueError("loss.guide.omega_cond needs 1 <= w_min <= w_max")
        if self.guide:
            if self.alpha > 0 or self.aux_on:
                raise ValueError("loss.guide is implemented for the BTM rule (signed alpha 0, no side heads)")
            if self.g_negative not in ("uncond", "rolled"):
                raise ValueError(f"loss.guide.negative must be uncond or rolled, got {self.g_negative}")
            if self.g_negative == "uncond" and self.label_dropout <= 0:
                raise ValueError("loss.guide.negative=uncond needs the null class trained (loss.label_dropout > 0)")
            if self.g_external:
                if not 0.0 <= self.g_tlo <= self.g_thi < 1.0:
                    raise ValueError(f"loss.guide interval [{self.g_tlo}, {self.g_thi}] must lie in [0, 1)")
            elif not 0.0 < self.g_tlo <= self.g_thi < 1.0 or 1.0 - self.anchor_alpha > self.g_tlo + 1e-9:
                raise ValueError(f"loss.guide interval [{self.g_tlo}, {self.g_thi}] must lie in (0, 1) and the anchors must reach "
                                 f"its start (1 - anchor_alpha = {1.0 - self.anchor_alpha:.3f} <= t_lo): the denoiser T(I_t; omega=t)")
        if self.aux_on and (not self.aux_pool or self.alpha > 0):
            raise ValueError("loss.aux needs aux.pool (one factor per model.aux_taps) and the BTM rule (signed alpha 0)")
        g = cfg.get("gcond") or {}
        self.gcond = bool(g.get("enabled", False))
        self.fr_grid = self.fr_bins = None
        if self.gcond:
            self.gw_form = str(g.get("form", "force"))
            self.gw_min, self.gw_max, self.gw_beta = float(g.get("w_min", 0.0)), float(g.w_max), float(g.beta)
            if self.gw_form not in ("force", "anchored"):
                raise ValueError(f"loss.gcond.form must be force or anchored, got {self.gw_form}")
            if not self.cfg_distill:
                raise ValueError("loss.gcond conditions the map on the CBM-E strength: it needs loss.cfg_distill.enabled")
            if not 0.0 <= self.gw_min <= self.gw_max:
                raise ValueError("loss.gcond needs 0 <= w_min <= w_max")
            if self.frechet is not None:
                grid = [float(v) for v in (g.get("frechet_w") or [])]
                if not grid:
                    raise ValueError("loss.gcond with the Frechet term on needs frechet_w, the w values it reads at")
                self.fr_grid = grid
                self.fr_bins = [self.frechet] + [make_fr() for _ in grid[1:]]
        if self.guide and self.gcond:
            raise ValueError("loss.guide and loss.gcond are separate modes; enable one")

    # ------------------------------------------------------------------ draws (a fixed order)
    def _time(self, x):
        t = torch.sigmoid(torch.randn(x.shape[0], device=x.device, dtype=x.dtype) * self.p_std + self.p_mean)
        return t * (1 - 2 * self.eps) + self.eps if self.eps > 0 else t

    def _omega(self, x):
        if self.anchor_alpha == 0.0:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        return 1.0 - self.anchor_alpha + self.anchor_alpha * torch.rand(x.shape[0], device=x.device, dtype=x.dtype)

    def _gw(self, x):
        """The guidance strength per example (loss.gcond): 1 + w has density prop. to (1 + w)^-beta on
        [1 + w_min, 1 + w_max], iMF's law for its CFG scale (beta 1 for B/2). A point law (w_min = w_max) draws nothing."""
        lo, hi, b = 1.0 + self.gw_min, 1.0 + self.gw_max, self.gw_beta
        if hi == lo:
            return torch.full((x.shape[0],), self.gw_min, device=x.device, dtype=x.dtype)
        u = torch.rand(x.shape[0], device=x.device, dtype=torch.float32)
        if b == 1.0:
            om = lo * (hi / lo) ** u
        else:
            a, c = lo ** (1.0 - b), hi ** (1.0 - b)
            om = (a + u * (c - a)) ** (1.0 / (1.0 - b))
        return (om - 1.0).clamp_min(0.0).to(x.dtype)

    @staticmethod
    def _interp(z, x, s):
        """(I_s, Idot_s) with alpha_s = 1 - s."""
        a = _bcast(1.0 - s, x)
        return a * z + (1.0 - a) * x, -1.0 * (z - x)

    def _anneal(self, step):
        if self.p_mean_end is not None and self.anneal_steps > 0:
            frac = min(1.0, max(0.0, (float(step or 0) - self.anneal_start) / self.anneal_steps))
            self.p_mean = self.p_mean0 + frac * (self.p_mean_end - self.p_mean0)

    # ------------------------------------------------------------------ the step (mode pretrain)
    def __call__(self, model, x, labels, step, teacher=None):
        """-> (weighted terms to be summed and backpropagated, aux scalars to log). `teacher` (loss.guide.teacher_decay):
        a slower copy (an EMA) that makes the fixed-kappa guide's reads instead of the online map."""
        labels = labels.long()
        self._anneal(step)
        if self.flow:
            return self._flow_step(model, x, labels)
        B = x.shape[0]
        z = torch.randn_like(x)
        if self.noise_scale != 1.0:
            z = z * self.noise_scale
        t = self._time(x)
        omega = self._omega(x)
        if self.restrict:            # transport time inside [0, omega], where the endpoint term does not pin T
            t = t * omega
        if self.t_mix > 0:           # drawn only when on, so the default draws are unchanged
            late = torch.rand_like(t) < self.t_mix
            t = torch.where(late, self.t_mix_lo + (omega - self.t_mix_lo).clamp_min(0.0) * torch.rand_like(t), t)
        u = torch.rand_like(omega)
        if self.omega_skew != 1.0:   # the anchor's target sharpens as omega' -> 1 (less of a conditional mean)
            u = 1.0 - (1.0 - u) ** self.omega_skew
        omega_prime = omega + (1.0 - omega) * u
        gw = self._gw(x) if self.gcond else None   # every term of an example sits on its own w slice
        input_labels = labels
        if self.label_dropout > 0:   # plain label dropout to the null class (off by default)
            drop = torch.rand(labels.shape, device=labels.device) < self.label_dropout
            labels = torch.where(drop, torch.full_like(labels, self.label_dim), labels)
        g_w = g_lo = g_hi = None
        if self.guide and self.g_omega_cond and int(step or 0) >= self.g_start:
            g_w, g_lo, g_hi = self._draw_omega(x)
        ikw = {} if g_lo is None else {"s_lo": g_lo, "s_hi": g_hi}
        I_t, v = self._interp(z, x, t)
        alpha = self.alpha
        aux = {} if g_w is None else {"guide_w_mean": g_w.float().mean().detach()}
        if self.ec and int(step or 0) >= self.g_start:   # before the guided reads: this step uses the last update's kappa
            aux.update(self._ec_step(model, x, input_labels))   # pre-dropout labels: every row has a class
        if alpha > 0:
            shift = 1 + int(torch.randint(B - 1, (1,), device="cpu"))   # a cyclic shift, never the identity
            order = torch.roll(torch.arange(B, device=x.device), shift)
            I_neg, v_neg = self._interp(z, x[order], t)
            neg_labels = labels[order] if self.paired else labels
            inputs, tangent = torch.cat((I_t, I_neg)), torch.cat((v, -v_neg))
            t_labels, t_omega = torch.cat((labels, neg_labels)), torch.cat((omega, omega))
            t_gw = None if gw is None else torch.cat((gw, gw))
            aux["signed_shift"] = self._scalar(shift, x.device)
        else:
            inputs, tangent, t_labels, t_omega, t_gw = I_t, v, labels, omega, (gw if gw is not None else g_w)
            if g_w is not None:
                tangent, g_aux, g_inside = self._guided_tangent_omega(model, I_t, v, t, labels, g_w, g_lo, g_hi)
                aux.update(g_aux)
            elif self.guide and int(step or 0) >= self.g_start:
                tangent, g_aux, g_inside = self._guided_tangent(model if teacher is None else teacher, I_t, v, t, labels, x)
                aux.update(g_aux)
        if self.aux_on:   # one multi-output JVP: the final map and every projected side readout
            def outputs(i):
                out, taps = model(i, t_labels, omega=t_omega, w=t_gw, return_aux=True)
                if len(taps) != len(self.aux_pool):
                    raise ValueError(f"model.aux_taps gives {len(taps)} side readouts, loss.aux.pool has {len(self.aux_pool)}")
                return (out, *(self._pool(q, k) for q, k in zip(taps, self.aux_pool)))
            outs, tans = torch.func.jvp(outputs, (inputs,), (tangent,))
            (pred, *aux_pred), (jvp, *aux_jvp) = outs, tans
        else:
            pred, jvp = torch.func.jvp(lambda i: model(i, t_labels, omega=t_omega, w=t_gw, **ikw), (inputs,), (tangent,))
        target = pred.detach() + jvp.detach()
        if self.guide and "guide_frac" in aux:   # the transport weight's spread where the guided tangent acts (a
            with torch.no_grad():                # concentrated weight starves the guidance of traction)
                wt = ((pred - target).float().flatten(1).pow(2).sum(1) + self.tc) ** -self.tp
                m = g_inside.float()
                n = m.sum().clamp_min(1.0)
                mean = (wt * m).sum() / n
                aux["guide_weight_cv"] = (((wt - mean).pow(2) * m).sum() / n).sqrt() / mean.clamp_min(1e-12)
        if alpha > 0:
            a = torch.full((B,), alpha, device=x.device, dtype=x.dtype)
            positive = (adaptive_mean(pred[:B], target[:B], self.tp, self.tc, sample_weight=1.0 + a) if self.p_within is None
                        else self._within_t_mean(pred[:B], target[:B], t, sample_weight=1.0 + a))
            transport = positive + adaptive_mean(pred[B:], target[B:], self.tp, self.tc, sample_weight=a)
            with torch.no_grad():
                sq = (pred - target).flatten(1).pow(2).sum(dim=1)
                aux.update(signed_positive=sq[:B].mean(), signed_negative=sq[B:].mean())
        elif self.p_within is not None:
            transport = self._within_t_mean(pred, target, t)
        else:
            transport = adaptive_mean(pred, target, self.tp, self.tc)
        diag = None
        e_omega = omega   # the endpoint rows' anchor input (the diagonal rows' own time when diag_time is on)
        if self.guide and self.g_diag_frac > 0:   # drawn only when on, so the default draws are unchanged
            diag = torch.rand_like(omega) < self.g_diag_frac
            if self.g_diag_time is not None:
                m, s = self.g_diag_time
                lo = 1.0 - self.anchor_alpha
                t_d = torch.sigmoid(m + s * torch.randn_like(omega)).clamp(lo, 1.0 - 1e-4)
                e_omega = torch.where(diag, t_d, omega)
                omega_prime = torch.where(diag, t_d, omega_prime)
            else:
                omega_prime = torch.where(diag, omega, omega_prime)
        I_end, _ = self._interp(z, x, omega_prime)
        end_target, e_w, e_kw = x, (gw if gw is not None else g_w), ikw
        if g_w is not None:   # diagonal rows are always guided and see the interval [0, 1] (iMF's r = t rows)
            e_lo, e_hi = g_lo, g_hi
            if diag is not None:
                e_lo, e_hi = torch.where(diag, torch.zeros_like(g_lo), g_lo), torch.where(diag, torch.ones_like(g_hi), g_hi)
            e_kw = {"s_lo": e_lo, "s_hi": e_hi}
            end_target, e_aux = self._guided_endpoint_omega(model, I_end, x, omega_prime, labels, g_w, e_lo, e_hi)
            aux.update(e_aux)
        elif self.guide and self.g_endpoint and int(step or 0) >= self.g_start:
            end_target, e_aux = self._guided_endpoint(model if teacher is None else teacher, I_end, x, omega_prime, labels)
            aux.update(e_aux)
        e_sw = self._late_weight(omega_prime)
        if self.aux_on:
            end_pred, end_taps = model(I_end, labels, omega=e_omega, w=e_w, return_aux=True, **e_kw)
            endpoint = adaptive_mean(end_pred, end_target, self.ep, self.ec, sample_weight=e_sw)
        else:
            endpoint = adaptive_mean(model(I_end, labels, omega=e_omega, w=e_w, **e_kw), end_target, self.ep, self.ec, sample_weight=e_sw)
        if e_sw is not None:
            aux["endpoint_late_weight_mean"] = e_sw.mean().detach()
        cfg_raw = self._cfg_distill_raw(model, I_t, t, pred[:B], labels, omega, gw) if self.cfg_distill else None
        cfg_scale = None
        if gw is not None:
            aux["gcond_w_mean"] = gw.float().mean().detach()
            if self.gw_form == "force":   # the term's norm is strength x mean(w) x |g_E|, shared out prop. to w
                m = (t <= self.cfg_t_max).float()
                cfg_scale = ((m * gw.float()).sum() / m.sum().clamp_min(1.0)).detach()
        w_t, w_e, bal_aux = self._balance(model, transport, endpoint)
        losses = {"transport": w_t * transport, "endpoint": w_e * endpoint}
        if self.transport_weight != 1.0:
            losses["transport"] = self.transport_weight * losses["transport"]
        if self.endpoint_weight != 1.0:
            losses["endpoint"] = self.endpoint_weight * losses["endpoint"]
        if self.diag_every and torch.is_grad_enabled() and int(step or 0) % self.diag_every == 0:
            aux.update(self._diagnostics(model, losses["transport"], losses["endpoint"]))
            if alpha > 0:
                aux["balance_transport_weight_x(1+2a)"] = (1.0 + 2.0 * alpha) * w_t
        aux.update(bal_aux, transport_unweighted=transport.detach(), endpoint_unweighted=endpoint.detach(),
                   interpolant_t_mean=t.mean().detach(), time_p_mean=self._scalar(self.p_mean, x.device))
        aux.update(self._weighted_terms(model, x, input_labels, endpoint, cfg_raw, step, losses, cfg_scale))
        if self.aux_on:
            aux.update(self._aux_terms(model, x, pred, aux_pred, aux_jvp, end_taps, losses))
        return losses, aux

    def _flow_step(self, model, x, labels):
        """loss.flow: MiniT2I's pretraining loss (minit2i-jax diffusion.py SimDDPM.forward, prediction x)."""
        if self.label_dropout > 0:   # the whole caption, as MiniT2I drops it
            drop = torch.rand(labels.shape, device=labels.device) < self.label_dropout
            labels = torch.where(drop, torch.full_like(labels, self.label_dim), labels)
        t = self._time(x)
        z = torch.randn_like(x) * self.noise_scale
        I_t, _ = self._interp(z, x, t)
        pred = model(I_t, labels, omega=t)   # the time-blind base (omega_tokens 0) ignores omega
        loss = ((pred.float() - x.float()) / _bcast((1.0 - t).clamp_min(self.flow_a_min), x)).pow(2).mean()
        return {"flow": loss}, {"interpolant_t_mean": t.mean().detach(), "time_p_mean": self._scalar(self.p_mean, x.device)}

    def _guided_tangent(self, model, I_t, v, t, labels, x=None):
        """w V + kappa b(I_t | y) + (1 - w - kappa) b(I_t | neg) inside the interval (non-null labels), V elsewhere; the
        velocities are the map's own denoiser read at omega = t, without gradient."""
        # t_cap (default 1: no cap) bounds the guided tangent only: the endpoint target stays guided up to t_hi (it has no
        # 1 / (1 - t)), so t_hi 0.9 with t_cap 0.6 guides the late endpoint band without the tangent's blow-up near t = 1
        inside = (t >= self.g_tlo) & (t <= min(self.g_thi, self.g_tcap)) & (labels != self.label_dim)
        with torch.no_grad():
            if self.g_negative == "uncond":
                neg = torch.full_like(labels, self.label_dim)
            else:
                shift = 1 + int(torch.randint(labels.shape[0] - 1, (1,), device="cpu"))
                neg = torch.roll(labels, shift)
            s = _bcast(1.0 - t, I_t)
            b_neg = (model(I_t, neg, omega=t) - I_t) / s
            guided = self.g_w * v + (1.0 - self.g_w - self.g_kappa) * b_neg
            den = b_y = None
            if self.g_kappa or self.g_vw != 1.0:
                den = model(I_t, labels, omega=t)
                b_y = (den - I_t) / s
                # kappa b_y, and (1 - v_weight) of the sample velocity swapped for the conditional read (v_weight 1: none)
                guided = guided + self.g_kappa * b_y + (1.0 - self.g_vw) * (b_y - v)
        # no_grad forwards inside autocast leave detached cast weights in the cache (see _cfg_distill_raw)
        torch.clear_autocast_cache()
        tangent = torch.where(_bcast(inside, v), guided.to(v.dtype), v)
        with torch.no_grad():
            m = inside.float()
            shift_sq = (tangent - v).float().flatten(1).pow(2).sum(1)
            aux = {"guide_frac": m.mean(), "guide_shift_rms": (shift_sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()}
            if den is not None and x is not None:   # the conditional denoiser T(I_t; y, omega = t) against the clean x
                err = (den - x).float().flatten(1).pow(2).mean(1)
                aux["guide_denoiser_mse"] = (err * m).sum().div(m.sum().clamp_min(1.0))
            if b_y is not None:   # the sample velocity's spread around the conditional read (what v_weight < 1 removes)
                vsq = (b_y - v).float().flatten(1).pow(2).sum(1)
                aux["guide_vnoise_rms"] = (vsq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()
        return tangent, aux, inside

    def _draw_omega(self, x):
        """iMF's guidance draws per example: w with density prop. to w^-beta on [w_min, w_max] (beta 1: w = w_min (w_max /
        w_min)^u), and the interval s_lo ~ U[0, 0.5], s_hi ~ U[0.5, 1]."""
        B, dev = x.shape[0], x.device
        u = torch.rand(B, device=dev, dtype=torch.float32)
        lo, hi, b = self.g_wmin, self.g_wmax, self.g_beta
        if hi == lo:
            w = torch.full((B,), lo, device=dev, dtype=torch.float32)
        elif b == 1.0:
            w = lo * (hi / lo) ** u
        else:
            a, c = lo ** (1.0 - b), hi ** (1.0 - b)
            w = (a + u * (c - a)) ** (1.0 / (1.0 - b))
        s_lo = 0.5 * torch.rand(B, device=dev, dtype=torch.float32)
        s_hi = 0.5 + 0.5 * torch.rand(B, device=dev, dtype=torch.float32)
        return w, s_lo, s_hi

    def _omega_reads(self, model, I, at, labels, w_g):
        """(D(I | y, w_g), D(I | null, 1)) at omega = at, interval [0, 1], no gradient (then the autocast cache cleared)."""
        full = {"s_lo": torch.zeros_like(at), "s_hi": torch.ones_like(at)}
        with torch.no_grad():
            d_y = model(I, labels, omega=at, w=w_g, **full)
            d_0 = model(I, torch.full_like(labels, self.label_dim), omega=at, w=torch.ones_like(w_g), **full)
        torch.clear_autocast_cache()
        return d_y, d_0

    def _guided_tangent_omega(self, model, I_t, v, t, labels, w, s_lo, s_hi):
        """iMF's guided tangent: V + (1 - 1/w_g)(D(y, w_g) - D(null, 1)) / (1 - t), w_g = w inside [s_lo, s_hi] (non-null
        labels) and 1 elsewhere, so the shift vanishes outside."""
        # below 1 - anchor_alpha the diagonal read T(I_t; omega = t) is untrained: no guidance there (a deviation from
        # iMF, whose interval can start at s = 0)
        inside = ((t >= s_lo.to(t.dtype)) & (t <= s_hi.to(t.dtype)) & (t >= 1.0 - self.anchor_alpha - 1e-6)
                  & (t <= self.g_tcap + 1e-6) & (labels != self.label_dim))
        w_g = torch.where(inside, w, torch.ones_like(w))
        d_y, d_0 = self._omega_reads(model, I_t, t, labels, w_g)
        with torch.no_grad():
            kap = _bcast((1.0 - 1.0 / w_g).to(v.dtype), v)
            tangent = v + kap * ((d_y - d_0) / _bcast(1.0 - t, I_t)).to(v.dtype)
            m = inside.float()
            shift_sq = (tangent - v).float().flatten(1).pow(2).sum(1)
            aux = {"guide_frac": m.mean(), "guide_shift_rms": (shift_sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()}
        return tangent, aux, inside

    def _guided_endpoint_omega(self, model, I_end, x, omega_prime, labels, w, s_lo, s_hi):
        """iMF's guided boundary: x + (1 - 1/w_g)(D(y, w_g) - D(null, 1)) for omega' inside [s_lo, s_hi], x elsewhere."""
        inside = ((omega_prime >= s_lo.to(omega_prime.dtype)) & (omega_prime <= s_hi.to(omega_prime.dtype))
                  & (omega_prime >= 1.0 - self.anchor_alpha - 1e-6) & (labels != self.label_dim))
        w_g = torch.where(inside, w, torch.ones_like(w))
        d_y, d_0 = self._omega_reads(model, I_end, omega_prime, labels, w_g)
        with torch.no_grad():
            target = x + _bcast((1.0 - 1.0 / w_g).to(x.dtype), x) * (d_y - d_0).to(x.dtype)
            m = inside.float()
            sq = (target - x).float().flatten(1).pow(2).sum(1)
            aux = {"guide_endpoint_frac": m.mean(),
                   "guide_endpoint_shift_rms": (sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()}
        return target, aux

    def _ec_apply(self):
        """kappa = 1 - 1/w for the tangent and the guided endpoint; the tangent's interval top inside the loop gain
        kappa / (1 - t_hi) <= 1.67 that R1d ran stably (self_signals.md section 2)."""
        w = math.exp(self.ec_logw)
        self.g_kappa = 1.0 - 1.0 / w
        self.g_thi = max(self.g_tlo, min(self.g_thi0, 1.0 - 0.6 * self.g_kappa))

    @torch.no_grad()
    def _ec_step(self, model, x, labels):
        """One evidence read (real vs one-step samples, same labels, s and noise) and, every `every` steps, one
        controller update; the host sync (all-reduce, .item()) only on update steps."""
        if self.ec_n > 0:
            y, xr = labels[: self.ec_n], x[: self.ec_n]
            n = y.shape[0]
            G = model(torch.randn_like(xr), y, omega=torch.full((n,), 0.8, device=x.device, dtype=x.dtype))
            s = self.ec_slo + (self.ec_shi - self.ec_slo) * torch.rand(n, device=x.device, dtype=x.dtype)
            eps = torch.randn_like(xr)
            tgt = torch.cat((xr, G.to(xr.dtype)))
            s2, e2 = torch.cat((s, s)), torch.cat((eps, eps))
            I = self._interp(e2, tgt, s2)[0]
            d_y = model(I, torch.cat((y, y)), omega=s2)
            d_0 = model(I, torch.full_like(torch.cat((y, y)), self.label_dim), omega=s2)
            torch.clear_autocast_cache()
            lam = (s2 / (1.0 - s2) ** 3).float()
            e = lam * ((d_0.float() - tgt.float()).flatten(1).pow(2).sum(1) - (d_y.float() - tgt.float()).flatten(1).pow(2).sum(1))
            v = torch.stack((e[:n].sum(), e[n:].sum(), torch.tensor(float(n), device=x.device)))
            self.ec_acc = v if self.ec_acc is None else self.ec_acc + v
        self.ec_count += 1
        if self.ec_count % self.ec_every == 0 and self.ec_acc is not None:
            v = self.ec_acc.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(v)
            r, g, n = (float(a) for a in v.tolist())
            self.ec_acc = None
            gap = (r - g) / max(n, 1.0)   # real - gen per row: > 0, the samples carry less class evidence than real data
            a = self.ec_ema
            if self.ec_gap is None:
                self.ec_gap, self.ec_var = gap, gap * gap
            else:
                self.ec_gap = a * self.ec_gap + (1 - a) * gap
                self.ec_var = a * self.ec_var + (1 - a) * (gap - self.ec_gap) ** 2
            z = self.ec_gap / max(self.ec_var, 1e-12) ** 0.5
            self.ec_logw = min(max(self.ec_logw + self.ec_eta * self.ec_every * z, math.log(self.ec_wlo)), math.log(self.ec_whi))
            self._ec_apply()
            self.ec_last = {"guide_ec_gap": gap, "guide_ec_gap_ema": self.ec_gap, "guide_ec_z": z}
        out = {"guide_w": math.exp(self.ec_logw), "guide_kappa": self.g_kappa, "guide_t_hi": self.g_thi, **self.ec_last}
        return {k: torch.tensor(float(val), device=x.device) for k, val in out.items()}

    def _late_weight(self, omega_prime):
        """loss.endpoint.late_weight: per-row weight min{((1 - t0) / (1 - omega'))^power, cap} above t0, 1 below; None when off."""
        if self.late_weight is None:
            return None
        t0, power, cap = self.late_weight
        op = omega_prime.float()
        w = ((1.0 - t0) / (1.0 - op).clamp_min(1e-6)).pow(power).clamp(max=cap)
        return torch.where(op > t0, w, torch.ones_like(w)).detach()

    def _guided_endpoint(self, model, I_end, x, omega_prime, labels):
        """The endpoint target with guidance (iMF's guided boundary): w x + kappa D(y) + (1 - w - kappa) D(neg) for
        omega' in [t_lo, t_hi] and non-null labels, x elsewhere; D(c) = T(I_omega'; c, omega = omega'), no gradient."""
        inside = (omega_prime >= self.g_tlo) & (omega_prime <= self.g_thi) & (labels != self.label_dim)
        with torch.no_grad():
            if self.g_negative == "uncond":
                neg = torch.full_like(labels, self.label_dim)
            else:
                shift = 1 + int(torch.randint(labels.shape[0] - 1, (1,), device="cpu"))
                neg = torch.roll(labels, shift)
            d_neg = model(I_end, neg, omega=omega_prime)
            guided = self.g_w * x + (1.0 - self.g_w - self.g_kappa) * d_neg
            if self.g_kappa:
                guided = guided + self.g_kappa * model(I_end, labels, omega=omega_prime)
        torch.clear_autocast_cache()
        target = torch.where(_bcast(inside, x), guided.to(x.dtype), x)
        with torch.no_grad():
            m = inside.float()
            sq = (target - x).float().flatten(1).pow(2).sum(1)
            aux = {"guide_endpoint_frac": m.mean(),
                   "guide_endpoint_shift_rms": (sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()}
        return target, aux

    @staticmethod
    def _pool(v, k):
        return F.avg_pool2d(v, k) if k > 1 else v

    def _aux_terms(self, model, x, pred, aux_pred, aux_jvp, end_taps, losses):
        """aux.alpha: the BTM rule on each projected side readout, balanced on its own readout; aux.align: agreement of
        each G_k with the projected final output on the same interpolant, both sides trained."""
        log, terms, K = {}, [], len(self.aux_pool)
        for k, (g, d, q_end, f) in enumerate(zip(aux_pred, aux_jvp, end_taps, self.aux_pool)):
            tr = adaptive_mean(g, g.detach() + d.detach(), self.tp, self.tc)
            en = adaptive_mean(self._pool(q_end, f), self._pool(x, f), self.ep, self.ec)
            w_t, w_e, _ = self._balance(model, tr, en, path=f"aux_heads.{k}")
            terms.append(w_t * tr + w_e * en)
            log.update({f"aux{k}_transport": tr.detach(), f"aux{k}_endpoint": en.detach()})
        if self.aux_alpha > 0:
            losses["aux_btm"] = (self.aux_alpha / K) * sum(terms)
        if self.aux_align > 0:
            align = sum((g.float() - self._pool(pred, f).float()).pow(2).mean() for g, f in zip(aux_pred, self.aux_pool)) / K
            losses["align"] = self.aux_align * align
            log["align_unweighted"] = align.detach()
        return log

    def _scalar(self, value, device):
        if self.sync_free:
            return torch.full((), float(value), device=device)
        return torch.as_tensor(float(value), device=device)

    def _within_t_mean(self, pred, target, t, sample_weight=None):
        """adaptive_mean with the weight split into a within-t part (d^2 + c)^-p_w and a per-bin part (m_t + c)^-(p - p_w)."""
        delta_sq = ((pred - target) ** 2).sum(dim=tuple(range(1, pred.ndim)))
        training = torch.is_grad_enabled()
        with torch.no_grad():
            m = self._bin_means(delta_sq.detach().float(), t, training).to(delta_sq.dtype)
            weight = (delta_sq.detach() + self.tc) ** -self.p_within * (m + self.tc) ** -(self.tp - self.p_within)
            if sample_weight is not None:
                weight = weight * sample_weight.detach().reshape(-1)
        return torch.mean((delta_sq * weight).float())

    def _t_bin(self, t):
        u = (torch.logit(t.detach().float(), eps=1e-6) + 4.0) / 8.0
        return (u * self.t_bins).floor().clamp(0, self.t_bins - 1).long()

    def _bin_means(self, delta_sq, t, training):
        """Per-example m_t. In training the bins' EMA is first updated from this step's bin sums, all-reduced over
        ranks. Outside training (validation) the state is only read, a bin never seen falls back to this batch's
        mean, and no collective runs."""
        bins = self._t_bin(t)
        stats = torch.zeros(2, self.t_bins, device=delta_sq.device, dtype=torch.float32)
        stats[0].index_add_(0, bins, delta_sq)
        stats[1].index_add_(0, bins, torch.ones_like(delta_sq))
        if self._bin_mean is None:
            mean = torch.zeros(self.t_bins, device=delta_sq.device)
            seen = torch.zeros(self.t_bins, device=delta_sq.device, dtype=torch.bool)
        else:
            mean, seen = self._bin_mean.to(delta_sq.device), self._bin_seen.to(delta_sq.device)
        if not training:
            return torch.where(seen, mean, stats[0] / stats[1].clamp_min(1.0))[bins]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
        has = stats[1] > 0
        batch = stats[0] / stats[1].clamp_min(1.0)
        updated = torch.where(seen, self.bin_ema * mean + (1.0 - self.bin_ema) * batch, batch)
        self._bin_mean, self._bin_seen = torch.where(has, updated, mean), seen | has
        return self._bin_mean[bins]

    def _diagnostics(self, model, transport, endpoint):
        """Full-network norms of the weighted terms' gradients, kappa_full = |g_E| / |g_T| and cos(g_T, g_E)."""
        params = tuple(q for q in model.parameters() if q.requires_grad)
        g_t = torch.autograd.grad(transport, params, retain_graph=True, allow_unused=True)
        g_e = torch.autograd.grad(endpoint, params, retain_graph=True, allow_unused=True)
        zero = torch.zeros((), device=transport.device, dtype=torch.float32)
        dot = sum(((a.float() * b.float()).sum() for a, b in zip(g_t, g_e) if a is not None and b is not None), zero)
        n_t = torch.sqrt(sum((a.float().pow(2).sum() for a in g_t if a is not None), zero))
        n_e = torch.sqrt(sum((b.float().pow(2).sum() for b in g_e if b is not None), zero))
        return {"diag_full_grad_norm_transport": n_t, "diag_full_grad_norm_endpoint": n_e,
                "diag_kappa_full": n_e / (n_t + 1e-12), "diag_cos_transport_endpoint": dot / (n_t * n_e + 1e-12)}

    # ------------------------------------------------------------------ state that rides in the checkpoint
    def state_dict(self):
        out = {}
        if self._bin_mean is not None:
            out["within_t"] = {"mean": self._bin_mean.detach().cpu(), "seen": self._bin_seen.detach().cpu()}
        if self._balance_norms is not None:
            out["balance_norms"] = tuple(n.detach().cpu() for n in self._balance_norms)
        if self.frechet is not None and self.fr_save_state:
            if self.fr_grid is not None:
                out["frechet_bins"] = {"w": list(self.fr_grid), "states": [b.state_dict() for b in self.fr_bins]}
            else:
                out["frechet"] = self.frechet.state_dict()
        return out

    def load_state_dict(self, state):
        if self.p_within is not None and "within_t" in state:
            self._bin_mean, self._bin_seen = state["within_t"]["mean"], state["within_t"]["seen"]
        if self.balance == "ema" and "balance_norms" in state:
            self._balance_norms = tuple(torch.as_tensor(n) for n in state["balance_norms"])
        if self.frechet is None or not self.fr_save_state:
            return
        if self.fr_grid is not None:
            # only this run's own per-w moments; a parent's single (w-free) state is not any slice's: fresh moments
            s = state.get("frechet_bins")
            if s is not None and [float(v) for v in s["w"]] == self.fr_grid:
                for b, st in zip(self.fr_bins, s["states"]):
                    b.load_state_dict(st)
        elif "frechet" in state:
            self.frechet.load_state_dict(state["frechet"])

    def _balance(self, model, transport, endpoint, path=None):
        """match_smaller_to_larger on the final layer: the smaller of the two gradient norms is scaled up to the larger."""
        one = torch.ones((), device=transport.device)
        if self.balance == "none":
            return one, one, {}
        module = model
        for part in (path or self.balance_path).split("."):
            module = getattr(module, part)
        params = tuple(p for p in module.parameters() if p.requires_grad)
        n_e, n_t = grad_norm(endpoint, params), grad_norm(transport, params)
        raw_e, raw_t = n_e.detach(), n_t.detach()
        if self.balance == "ema" and path is None:   # the smoothed norms set the weights; the raw ones are still logged
            if self._balance_norms is None:
                self._balance_norms = (raw_e.clone(), raw_t.clone())
            else:
                d, (m_e, m_t) = self.balance_ema, (n.to(raw_e.device) for n in self._balance_norms)
                self._balance_norms = (d * m_e + (1 - d) * raw_e, d * m_t + (1 - d) * raw_t)
            n_e, n_t = self._balance_norms
        w_e = torch.where(n_e < n_t, torch.clamp_min(n_t / (n_e + 1e-8), 1.0), one).detach()
        w_t = torch.where(n_t < n_e, torch.clamp_min(n_e / (n_t + 1e-8), 1.0), one).detach()
        return w_t, w_e, {"balance_transport_weight": w_t, "balance_endpoint_weight": w_e,
                          "final_layer_grad_norm_transport": raw_t, "final_layer_grad_norm_endpoint": raw_e}

    def _cfg_distill_raw(self, model, I_t, t, pred, labels, omega, gw=None):
        """CBM-E, unweighted: target sg(2 T(I_t, y) - T(I_t, y^-)) on the transport forward's own draws with t <= t_max.

        With loss.gcond (gw = each example's w; pred, neg at that w):
            force     the same target, the example's term times w (w = 1 is the recipe; the push has no fixed point, so a
                      scale inside the target would be cancelled by the p = 1 bound and the norm matching: w is a force)
            anchored  target sg(T(I_t, y; 0) + w / (1 + w) (T(I_t, y; w) - T(I_t, y^-; w))), iMF's guided target on the
                      map: fixed point T(y; w) = T(y; 0) + w (T(y; 0) - mean_y' T(y'; 0)), CFG of the w = 0 slice."""
        B = I_t.shape[0]
        shift = 1 + int(torch.randint(B - 1, (1,), device="cpu"))
        neg_labels = labels[torch.roll(torch.arange(B, device=I_t.device), shift)]
        mask = (t <= self.cfg_t_max).to(pred.dtype)
        anchored = gw is not None and self.gw_form == "anchored"
        with torch.no_grad():
            neg = model(I_t, neg_labels, omega=omega, w=gw)
            base = model(I_t, labels, omega=omega, w=torch.zeros_like(gw)) if anchored else None
        # a no_grad forward inside autocast leaves detached cast weights in the cache; the grad forwards after it would
        # then reach only the fp32 parameters (measured: the matrices' update ~10x too small). Clear it.
        torch.clear_autocast_cache()
        if anchored:
            kappa = _bcast(gw.float() / (1.0 + gw.float()), pred)
            target = base.float() + kappa * (pred.detach().float() - neg.float())
        else:
            target = (2.0 * pred.detach() - neg.detach()).float()
        per = (pred.float() - target).pow(2).flatten(1).sum(dim=1)
        bound = 1.0 / (per.detach() + self.tc) ** self.tp        # the transport branches' per-example bound
        weight = mask.float() * gw.float() if (gw is not None and not anchored) else mask.float()
        return (per * bound * weight).sum() / mask.float().sum().clamp_min(1.0)

    def _weighted_terms(self, model, x, labels, endpoint, cfg_raw, step, losses, cfg_scale=None):
        """CBM-E and the Frechet term, each scaled to a fraction of the endpoint's full-parameter gradient norm.
        cfg_scale (loss.gcond, form force): the batch's mean w over the CBM-E examples, multiplying CBM-E's norm.
        With loss.gcond the Frechet batch of step s reads at w = frechet_w[s mod K], with that slice's own moments."""
        aux = {}
        params = tuple(q for q in model.parameters() if q.requires_grad)
        end_norm = None
        if cfg_raw is not None and self.cfg_fixed_weight is not None:
            w = torch.full((), self.cfg_fixed_weight, device=cfg_raw.device)
            losses["cfg_distill"] = w * cfg_raw
            aux.update(cfg_distill_unweighted=cfg_raw.detach(), cfg_distill_weight=w)
        elif cfg_raw is not None:
            end_norm = grad_norm(endpoint, params).detach()
            cfg_norm = grad_norm(cfg_raw, params).detach()
            w = (self.cfg_strength * end_norm / (cfg_norm + 1e-8)).detach()
            if cfg_scale is not None:
                w = w * cfg_scale
            losses["cfg_distill"] = w * cfg_raw
            aux.update(cfg_distill_unweighted=cfg_raw.detach(), cfg_distill_weight=w)
        if self.frechet is not None and torch.is_grad_enabled() and int(step or 0) >= self.fr_start:
            n = min(self.fr_n, x.shape[0])
            z = torch.randn_like(x[:n])
            term, kw, tag = self.frechet, {}, ""
            if self.fr_grid is not None:
                k = int(step or 0) % len(self.fr_grid)
                term, tag = self.fr_bins[k], f"_w{self.fr_grid[k]:g}"
                kw = {"w": torch.full((n,), self.fr_grid[k], device=x.device, dtype=x.dtype)}
                if term._nets is None:        # one VAE + DINO for every slice
                    term._nets = next((b._nets for b in self.fr_bins if b._nets is not None), None)
            read = model(z, labels[:n], omega=torch.full((n,), self.fr_omega, device=x.device, dtype=x.dtype), **kw)
            if end_norm is None and self.fr_fixed_weight is None:
                end_norm = grad_norm(endpoint, params).detach()
            fr_loss, fr_aux = term(read)
            grads = torch.autograd.grad(fr_loss, params, allow_unused=True)
            if self.fr_global_norm and dist.is_available() and dist.is_initialized():
                flat = torch.cat([g.detach().float().flatten() for g in grads if g is not None])
                dist.all_reduce(flat, op=dist.ReduceOp.AVG)
                fr_norm = flat.norm()
                del flat
            else:
                fr_norm = torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads if g is not None))
            anneal = fr_aux["frechet_anneal"] if self.fr_anneal else torch.ones_like(fr_aux["frechet_anneal"])
            if self.fr_fixed_weight is not None:   # w on the unit loss's gradient, times the anneal if it is on
                w = (self.fr_fixed_weight * anneal).detach()
            else:
                w = (self.fr_strength * end_norm * anneal / (fr_norm + 1e-8)).detach()
            aux["frechet_grad_norm"] = fr_norm.detach()
            # the graph through the frozen decoder + DINO is walked once; a linear surrogate carries w * g into the step
            surrogate = sum((q * (w * g).detach().to(q.dtype)).sum() for q, g in zip(params, grads) if g is not None)
            losses["frechet"] = surrogate - surrogate.detach() + w * fr_loss.detach()
            fr_aux = {**fr_aux, "frechet_unweighted": fr_loss.detach(), "frechet_weight": w}
            aux.update({f"{key}{tag}": v for key, v in fr_aux.items()})
        return aux


# ---------------------------------------------------------------------- mode finetune
def draw_pairs(targets, seed, step, *, pairs, offset=0, time_mean=-1.2, time_std=1.0, omega_range=(0.6, 1.0)):
    """The per-pair draw stream: pair k = pairs * step + offset + p has its own generator, seed + 7000001 k, and
    draws, in order, z (fp32), the time's normal, omega ~ U[omega_range], the endpoint anchor omega' ~ U[omega, 1]."""
    out = {"source": [], "time": [], "omega": [], "anchor": []}
    for p in range(len(targets) // 2):
        g = torch.Generator(device=targets.device).manual_seed(seed + 7000001 * (step * pairs + offset + p))
        out["source"].append(torch.randn(targets[2 * p:2 * p + 2].shape, device=targets.device, dtype=torch.float32, generator=g))
        n = torch.randn(2, device=targets.device, dtype=torch.float32, generator=g)
        out["time"].append(1e-4 + (1 - 2e-4) * torch.sigmoid(n * time_std + time_mean))
        omega = omega_range[0] + (omega_range[1] - omega_range[0]) * torch.rand(2, device=targets.device, generator=g)
        out["omega"].append(omega)
        out["anchor"].append(omega + (1 - omega) * torch.rand(2, device=targets.device, generator=g))
    return {k: torch.cat(v) for k, v in out.items()}


def _pair_branch(model, x, labels, d, tp, tc, ep, ec, negate, guide=None, aux=None):
    """One branch: I = (1 - t) z + t x, tangent x - z (negated for the signed branch).

    guide (positive branch only): iMF's rule on the fine-tune. For t in [t_lo, t_hi] and
    non-null labels the tangent is V + kappa (b_y - b_null), b_c = (T(I_t; c, omega = t) - I_t) / (1 - t) the map's own
    denoiser velocity (no gradient); for anchors omega' in [t_lo, t_hi] the endpoint target is x + kappa (D_y - D_null),
    D_c = T(I_omega'; c, omega = omega'). Needs the null class trained (label dropout > 0) and anchors
    reaching t_lo (finetune.omega_range starting at <= t_lo)."""
    # the draws stay fp32 (the bank latents are bf16), so the interpolants are formed in fp32
    t, a = _bcast(d["time"].to(x.device), x), _bcast(d["anchor"].to(x.device), x)
    z = d["source"].to(x.device)
    I_t = (1 - t) * z + t * x
    v = x - z
    tangent = (-v) if negate else v
    if guide is not None and not negate:
        null = torch.full_like(labels, guide["null"])
        tf = d["time"].to(x.device)
        inside = (tf >= guide["t_lo"]) & (tf <= guide["t_hi"]) & (labels != guide["null"])
        with torch.no_grad():
            s = 1.0 - t
            b_y = (model(I_t, labels, omega=tf) - I_t) / s
            b_0 = (model(I_t, null, omega=tf) - I_t) / s
            guided = v + guide["kappa"] * (b_y - b_0)
        torch.clear_autocast_cache()
        tangent = torch.where(_bcast(inside, v), guided.to(v.dtype), v)
        if aux is not None:
            m = inside.float()
            aux["guide_frac"] = float(m.mean())
            aux["guide_shift_rms"] = float(((tangent - v).float().flatten(1).pow(2).sum(1) * m).sum().div(m.sum().clamp_min(1)).sqrt())
    pred, jvp = torch.func.jvp(lambda i: model(i, labels, omega=d["omega"]), (I_t,), (tangent,))
    transport = pair_means(pred, (pred.detach() + jvp.detach()).detach(), tp, tc)
    if negate:
        return transport, None
    I_end = (1 - a) * z + a * x
    end_target = x
    if guide is not None and guide.get("endpoint", True):
        af = d["anchor"].to(x.device)
        inside_e = (af >= guide["t_lo"]) & (af <= guide["t_hi"]) & (labels != guide["null"])
        with torch.no_grad():
            d_y = model(I_end, labels, omega=af)
            d_0 = model(I_end, torch.full_like(labels, guide["null"]), omega=af)
            g_end = x + guide["kappa"] * (d_y - d_0)
        torch.clear_autocast_cache()
        end_target = torch.where(_bcast(inside_e, x), g_end.to(x.dtype), x)
        if aux is not None:
            aux["guide_endpoint_frac"] = float(inside_e.float().mean())
    return transport, pair_means(model(I_end, labels, omega=d["omega"]), end_target, ep, ec)


def pair_terms(model, targets, labels, draws, *, alpha, paired, generator, transport_pc, endpoint_pc, guide=None):
    """The fine-tuning recipe's two unweighted terms on one micro-batch, per pair: transport = (1 + alpha) T+ + alpha T-,
    endpoint on the positive branch only (the caller takes .mean()). `generator` (CPU) draws the roll's shift, continuing the stream that drew the
    bank rows. Autocast (bf16 on GPU) is the caller's."""
    (tp, tc), (ep, ec) = transport_pc, endpoint_pc
    g_aux = {}
    pos_t, pos_e = _pair_branch(model, targets, labels, draws, tp, tc, ep, ec, negate=False, guide=guide, aux=g_aux)
    if not alpha:
        return {"transport": pos_t, "endpoint": pos_e}, g_aux
    B = len(targets)
    shift = 1 + int(torch.randint(B - 1, (1,), generator=generator, device="cpu"))
    order = torch.roll(torch.arange(B, device=targets.device), shift)
    neg_t, _ = _pair_branch(model, targets[order], labels[order] if paired else labels, draws, tp, tc, ep, ec, negate=True)
    return ({"transport": (1.0 + alpha) * pos_t + alpha * neg_t, "endpoint": pos_e},
            {"signed_positive": float(pos_t.detach().float().mean()), "signed_negative": float(neg_t.detach().float().mean()),
             "signed_shift": shift, **g_aux})
