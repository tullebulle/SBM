"""The self-guided objective (loss.name=guided_cbm) for the RAE-latent ImageNet models and the world models.

It follows the reference implementation of the SD-VAE models (src/sbm/loss.py) step by step: the random draws are made
in the same order, so with `time_shift: 1` and `balance.mode: none` it returns the same loss values
(tests/rae/test_guided_cbm.py checks this).

Notation (as in the reference code): I_t = (1 - t) z + t x, t = 1 data, V = x - z; T(I; y, omega) the map, omega the anchor (network input),
y the class or the null class (index num_classes).

    times      t ~ logit-normal(p_mean, p_std) (eps-clipped); omega ~ U[1 - anchor_alpha, 1]; with probability mix_frac
               t is redrawn U[mix_lo, omega) (the late path); omega' ~ U[omega, 1], set to omega on diag_frac of the rows
    labels     dropped to the null class with probability label_dropout (trains the CFG reference)
    transport  || T(I_t; y, omega) - sg(T + J_T(I_t) V~) ||^2 * sg[(d^2 + c)^-p]  (one JVP), with the guided tangent
               V~ = V + kappa (b(y) - b(null)) for t in [t_lo, t_hi] and y not null, V elsewhere;
               b(c) = (T_teacher(I_t; c, omega = t) - I_t) / (1 - t), the teacher's denoiser velocity, no gradient
    endpoint   || T(I_omega'; y, omega) - X~ ||^2 * late(omega'),  X~ = x + kappa (D(y) - D(null)) for omega' in
               [t_lo, t_hi] and y not null, x elsewhere; D(c) = T_teacher(I_omega'; c, omega = omega');
               late(omega') = min{((1 - t0) / (1 - omega'))^power, cap} for omega' > t0, 1 below
    total      transport_weight * transport + endpoint_weight * endpoint
    teacher    a frozen copy of the active EMA, refreshed every teacher_refresh steps (held by the trainer, which
               sets `self.teacher` before each step; sbm_rae.trainer)

Representation alignment (REPA; RAEv2, arXiv 2605.18324, finds it complementary to RAE latents), off by default:

    repa               the endpoint forward also returns the trunk's block-k tokens through a training-only MLP projector
                       (model.repa_enabled, repa_encoder_depth, repa_z_dim = the latent's channels); they are pulled
                       toward the clean latent's tokens by cosine similarity (the RAE latent is the encoder's last layer,
                       so no second encoder is needed), with `weight` relative to a per-element-mean squared error.

One addition from iMF (Geng et al., arXiv 2512.02012, Sec. 4.1), off by default:

    guide.marginal_tangent   from the guide start the transport JVP runs along b(y), the teacher's marginal-velocity
                       estimate, instead of the conditional V = x - z (which the JVP amplifies; iMF's loss-variance point),
                       for nominal t in [t_lo, t_hi] (t_lo defaults to 1 - anchor_alpha: the denoiser read at anchor t is
                       trained only there; t_hi caps the 1 / (1 - t_a) amplification of the denoiser's error). Inside the
                       guide band the tangent is b(y) + kappa (b(y) - b(null)). No extra forward: b(y) is already read.

Two additions for a new latent space, both identities at their defaults:

    time_shift alpha   every time above is a NOMINAL time s; the interpolant sits at the actual time
                       s_a = s / (s + alpha (1 - s)) (RAE's dimension shift: a high-dimensional latent keeps more signal at
                       the same noise level, so the same nominal schedule must hold more noise). The network is
                       conditioned on the nominal anchor, the guide interval, the late band and the time law are nominal,
                       and the denoiser velocity divides by the actual 1 - t_a (it is the straight line's velocity).
                       alpha = 1 is the reference algorithm.
    balance.mode       none: the reference fixed weights (transport.weight, endpoint.weight).
                       freeze: until guide.start_step the final-layer gradient balancer match_smaller_to_larger (the reference
                       from-scratch recipe), tracking an EMA of |g_E| / |g_T|; from start_step the transport weight is
                       that EMA, all-reduced once and frozen (the reference fixed-weight form, with the 2000 measured on SD-VAE
                       latents re-measured in this latent space). The frozen value rides in the checkpoint.
"""

import os

import torch
import torch.distributed as dist

# The balancer's _grad_norm calls torch.autograd.grad(..., retain_graph=True) through the torch.compile'd student forward;
# since torch 2.9 the compiled backward donates its buffers and refuses retain_graph=True ("This backward function was
# compiled with non-empty donated buffers ..."), e.g. after a recompile on a partial batch. Documented workaround: costs
# some memory, changes no numbers.
try:
    import torch._functorch.config as _functorch_config

    _functorch_config.donated_buffer = False
except Exception:  # pragma: no cover - older torch without the flag
    pass


def _bcast(t, like):
    return t.view(-1, *((1,) * (like.ndim - 1)))


def adaptive_mean(pred, target, p, c, sample_weight=None):
    """mean_b d_b^2 * sg[(d_b^2 + c)^-p] (* sample_weight_b), d_b^2 summed over the example; p = 0 is plain L2.
    The product is formed in the prediction's dtype (bf16 under autocast) and averaged in fp32 (the reference reduction)."""
    delta_sq = ((pred - target) ** 2).sum(dim=tuple(range(1, pred.ndim)))
    weight = (1.0 / (delta_sq + c) ** p).detach()
    if sample_weight is not None:
        weight = weight * sample_weight.detach().reshape(-1)
    return torch.mean((delta_sq * weight).float())


def _grad_norm(loss, parameters):
    out = torch.zeros((), device=loss.device, dtype=torch.float32)
    if not loss.requires_grad or not parameters:
        return out
    for g in torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True):
        if g is not None:
            out += g.detach().float().pow(2).sum()
    return torch.sqrt(out)


def load_weak_state(path, ema=None):
    """The model weights (keys without the "model." prefix) of a training checkpoint for autoguidance's weak model: its
    active EMA (the checkpoint's state_dict, which sbm_rae.ema swaps in at save; what the teacher is copied from) when
    `ema` is None, else the EMA copy with that decay (ema_state_dicts, e.g. 0.9999)."""
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    active = ck.get("ema_active_decay", None)
    if ema is None or (active is not None and abs(float(ema) - float(active)) < 1e-12):
        sd = ck["state_dict"]
    else:
        label = f"{float(ema):g}".replace(".", "_")
        stored = ck.get("ema_state_dicts", {})
        if label not in stored:
            raise KeyError(f"{path}: no EMA {ema} (stored: {sorted(stored)}, active: {active})")
        sd = stored[label]
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


def guidance_scale_cond(gcfg):
    """(enabled, lo, hi, w0, a) of loss.guided_cbm.guide.scale_cond (`gcfg` = the loss.guided_cbm block): the guide's
    scale w0 = 1 + kappa / (1 - kappa - weight) and the autoguidance ratio a = weight / (1 - kappa - weight), from
    guide.kappa and the autoguide weight (0 without autoguidance). Read by the trainer and the scorer too."""
    gd = gcfg.get("guide", None) or {}
    sc = gd.get("scale_cond", None) or {}
    enabled = bool(sc.get("enabled", False))
    kappa = float(gd.get("kappa", 0.8))
    ag = gd.get("autoguide", None) or {}
    lam = float(ag.get("weight", 0.0) or 0.0) if bool(ag.get("enabled", False)) else 0.0
    rest = 1.0 - kappa - lam
    if enabled and rest <= 0:
        raise ValueError(f"loss.guided_cbm.guide.scale_cond needs kappa + autoguide weight < 1, got {kappa} + {lam}")
    w0 = 1.0 + kappa / rest if rest > 0 else float("inf")
    a = lam / rest if rest > 0 else 0.0
    return enabled, float(sc.get("lo", 2.0)), float(sc.get("hi", 8.0)), w0, a


def scale_cond_lr_mult(cfg):
    """loss.guided_cbm.guide.scale_cond.lr_mult of a run config (`cfg` = the root): the learning-rate multiplier of the
    network's guidance-scale embedding; 1 unless the run is guided_cbm with scale_cond on. Read by the trainer."""
    loss = cfg.get("loss", None) or {}
    if loss.get("name") != "guided_cbm" or loss.get("guided_cbm", None) is None:
        return 1.0
    sc = (loss.guided_cbm.get("guide", None) or {}).get("scale_cond", None) or {}
    if not bool(sc.get("enabled", False)):
        return 1.0
    mult = float(sc.get("lr_mult", 1.0))
    if mult <= 0:
        raise ValueError(f"loss.guided_cbm.guide.scale_cond.lr_mult must be > 0, got {mult}")
    return mult


class GuidedCBMLoss:
    def __init__(self, cfg, label_dim):
        """cfg: the `loss.guided_cbm` block (configs/rae/train.yaml)."""
        self.label_dim = int(label_dim)
        self.time_shift = float(cfg.get("time_shift", 1.0))
        if self.time_shift <= 0:
            raise ValueError(f"loss.guided_cbm.time_shift must be > 0, got {self.time_shift}")
        tr, en = cfg.transport, cfg.endpoint
        self.tp, self.tc = float(tr.p), float(tr.c)
        self.transport_weight = float(tr.get("weight", 1.0))
        self.ep, self.ec = float(en.p), float(en.c)
        self.endpoint_weight = float(en.get("weight", 1.0))
        lw = en.get("late_weight")
        self.late_weight = None if not lw else (float(lw["t0"]), float(lw.get("power", 1.0)), float(lw.get("cap", 8.0)))
        self.anchor_alpha = float(cfg.anchor_alpha)
        t = cfg.time
        self.p_mean, self.p_std, self.eps = float(t.p_mean), float(t.p_std), float(t.get("eps", 1e-4))
        self.t_mix, self.t_mix_lo = float(t.get("mix_frac", 0.0)), float(t.get("mix_lo", 0.3))
        self.label_dropout = float(cfg.get("label_dropout", 0.0))
        b = cfg.balance
        self.balance = str(b.get("mode", "none"))
        if self.balance not in ("none", "freeze"):
            raise ValueError(f"loss.guided_cbm.balance.mode must be none or freeze, got {self.balance}")
        self.balance_path = str(b.get("parameter_path", "final_layer"))
        self.balance_ema = float(b.get("ema", 0.999))
        self._ratio_ema = None        # EMA of |g_E| / |g_T| on the final layer (mode freeze, before the freeze)
        self._frozen_weight = None    # the transport weight from the freeze on (mode freeze)
        gd = cfg.guide
        self.guide = bool(gd.get("enabled", True))
        self.g_kappa = float(gd.get("kappa", 0.8))
        self.g_tlo, self.g_thi = float(gd.get("t_lo", 0.35)), float(gd.get("t_hi", 0.6))
        self.g_start = int(gd.get("start_step", 0))
        self.g_endpoint, self.g_diag_frac = bool(gd.get("endpoint", True)), float(gd.get("diag_frac", 0.5))
        self.teacher_refresh = int(gd.get("teacher_refresh", 10000) or 0)
        # which EMA copy the teacher is copied from (null: the active EMA, the reference recipe), and whether a resumed or
        # initialised run refreshes it at its first guided step (a continuation that changes the source)
        ts = gd.get("teacher_source", None)
        self.teacher_source = None if ts is None else float(ts)
        self.teacher_refresh_at_start = bool(gd.get("teacher_refresh_at_start", False))
        mt = gd.get("marginal_tangent", None) or {}
        self.g_marginal = bool(mt.get("enabled", False))
        mlo = mt.get("t_lo", None)
        self.m_tlo = 1.0 - self.anchor_alpha if mlo is None else float(mlo)
        self.m_thi = float(mt.get("t_hi", 0.95))
        # autoguidance (off by default): + weight (b(y) - b_weak(y)) on the guided band, b_weak from a frozen weaker
        # checkpoint the trainer loads into self.weak (sbm_rae.guided_cbm.load_weak_state)
        ag = gd.get("autoguide", None) or {}
        self.ag = bool(ag.get("enabled", False))
        self.ag_weight = float(ag.get("weight", 0.0) or 0.0)
        self.ag_ckpt = ag.get("ckpt", None)
        self.ag_ema = None if ag.get("ema", None) is None else float(ag.get("ema"))
        if self.ag and not (self.guide and self.ag_weight > 0 and self.ag_ckpt):
            raise ValueError("loss.guided_cbm.guide.autoguide needs the guide on, weight > 0 and a ckpt")
        if self.ag and self.g_kappa + self.ag_weight >= 1.0:
            print(f"[guided_cbm] warning: kappa {self.g_kappa} + autoguide weight {self.ag_weight} >= 1: the self-referential "
                  "teacher compounds both, so the guidance has no finite fixed point", flush=True)
        self.weak = None              # the frozen weak model (autoguide), set by the trainer
        # guidance-scale conditioning (off by default): a per-example scale w as a network input, and the guide's
        # kappa / autoguide weight following it (guidance_scale_cond)
        self.scale_cond, self.sc_lo, self.sc_hi, self.sc_w0, self.sc_a = guidance_scale_cond(cfg)
        if self.scale_cond and not (self.guide and 1.0 <= self.sc_lo <= self.sc_hi):
            raise ValueError(f"loss.guided_cbm.guide.scale_cond needs the guide on and 1 <= lo <= hi, got "
                             f"[{self.sc_lo}, {self.sc_hi}]")
        if self.g_marginal and not (1.0 - self.anchor_alpha - 1e-9 <= self.m_tlo <= self.m_thi < 1.0):
            raise ValueError(f"guide.marginal_tangent needs 1 - anchor_alpha <= t_lo <= t_hi < 1 (the denoiser read at anchor t "
                             f"is trained only there), got [{self.m_tlo}, {self.m_thi}]")
        lws = cfg.endpoint.get("late_weight_start_step", None)
        # the late band is weighted from the guide start by default: the reference late band runs with fixed weights only (with
        # the adaptive balancer, a large cap collapses training)
        self.late_start = self.g_start if lws is None else int(lws)
        self.tangent_only_jvp = bool(cfg.get("tangent_only_jvp", True))
        rp = cfg.get("repa", None) or {}
        self.repa = bool(rp.get("enabled", False))
        self.repa_weight = float(rp.get("weight", 0.5))
        # phi (feature-space loss terms) is not included in this code
        if bool((cfg.get("phi", None) or {}).get("enabled", False)):
            raise NotImplementedError("loss.guided_cbm.phi (feature-space terms) is not included in this code; "
                                      "set loss.guided_cbm.phi.enabled=false")
        self.teacher = None           # set by the trainer before each step from guide.start_step on
        # speed (trainer.compile.*, sbm_rae.speed): optional replacements for the student's two calls, e.g. torch.compile'd;
        # None runs the stock eager calls below
        self.student_jvp = None       # (model, I, tangent, labels, omega) -> (pred, jvp)
        self.student_fwd = None       # (model, I, labels, omega, return_repa=False) -> pred, or (pred, projection)
        if self.guide:
            if self.label_dropout <= 0:
                raise ValueError("loss.guided_cbm.guide needs the null class trained (label_dropout > 0)")
            if not 0.0 < self.g_tlo <= self.g_thi < 1.0 or 1.0 - self.anchor_alpha > self.g_tlo + 1e-9:
                raise ValueError(f"guide interval [{self.g_tlo}, {self.g_thi}] must lie in (0, 1) and the anchors must reach "
                                 f"its start (1 - anchor_alpha = {1.0 - self.anchor_alpha:.3f} <= t_lo)")
        if self.balance == "freeze" and not self.guide:
            raise ValueError("loss.guided_cbm.balance.mode freeze freezes at guide.start_step: it needs the guide")

    # ------------------------------------------------------------------ what the trainer reads
    def teacher_schedule(self):
        """(first step, refresh period) of the frozen teacher, or None when no step reads it."""
        if not self.guide:
            return None
        return self.g_start, self.teacher_refresh

    def state_dict(self):
        r = self._ratio_ema
        return {"ratio_ema": None if r is None else float(r), "frozen_weight": self._frozen_weight}

    def load_state_dict(self, state):
        r = state.get("ratio_ema")
        self._ratio_ema = None if r is None else torch.tensor(float(r), dtype=torch.float32)
        self._frozen_weight = state.get("frozen_weight")

    # ------------------------------------------------------------------ the condition (class labels here; a dict for the
    # Bridge world model, sbm_rae.bridge_loss overrides these four)
    def _cast_labels(self, labels):
        return labels.long()

    def _drop_labels(self, labels, drop):
        """labels with the rows of `drop` (B,) bool set to the null class."""
        return torch.where(drop, torch.full_like(labels, self.label_dim), labels)

    def _null_mask(self, labels):
        """(B,) bool: which rows carry the null condition."""
        return labels == self.label_dim

    def _null_labels(self, labels):
        """the null condition for every row (the guide's unconditional read)."""
        return torch.full_like(labels, self.label_dim)

    # ------------------------------------------------------------------ draws (the reference order)
    def _warp(self, s):
        """nominal -> actual interpolant time, s / (s + alpha (1 - s)); the identity at alpha 1."""
        if self.time_shift == 1.0:
            return s
        return s / (s + self.time_shift * (1.0 - s))

    def _time(self, x):
        t = torch.sigmoid(torch.randn(x.shape[0], device=x.device, dtype=x.dtype) * self.p_std + self.p_mean)
        return t * (1 - 2 * self.eps) + self.eps if self.eps > 0 else t

    def _omega(self, x):
        if self.anchor_alpha == 0.0:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        return 1.0 - self.anchor_alpha + self.anchor_alpha * torch.rand(x.shape[0], device=x.device, dtype=x.dtype)

    @staticmethod
    def _interp(z, x, s):
        """(I_s, Idot_s) as the reference forms them (alpha_s = 1 - s)."""
        a = _bcast(1.0 - s, x)
        return a * z + (1.0 - a) * x, -1.0 * (z - x)

    # ------------------------------------------------------------------ the step
    def __call__(self, model, weighting_model, x, labels, global_step, repa_targets=None):
        del weighting_model, repa_targets
        step = int(global_step or 0)
        labels = self._cast_labels(labels)
        z = torch.randn_like(x)
        t = self._time(x)
        omega = self._omega(x)
        if self.t_mix > 0:           # half the transport times on the late path [mix_lo, omega)
            late = torch.rand_like(t) < self.t_mix
            t = torch.where(late, self.t_mix_lo + (omega - self.t_mix_lo).clamp_min(0.0) * torch.rand_like(t), t)
        u = torch.rand_like(omega)
        omega_prime = omega + (1.0 - omega) * u
        if self.label_dropout > 0:
            drop = torch.rand(x.shape[0], device=x.device) < self.label_dropout
            labels = self._drop_labels(labels, drop)
        guided = self.guide and step >= self.g_start
        reader = model if self.teacher is None else self.teacher
        if guided and self.teacher is None and self.teacher_refresh and torch.is_grad_enabled():
            # (a validation pass before any teacher exists reads the map itself)
            raise RuntimeError("loss.guided_cbm: the guided step needs the trainer's frozen teacher (loss.teacher)")
        w = None
        if self.scale_cond:   # the network's guidance-scale input: drawn per example once the guide is on, w0 before
            w = torch.full_like(omega, self.sc_w0)
            if guided:   # (a fixed scale, lo = hi, draws nothing)
                w = (torch.full_like(omega, self.sc_lo) if self.sc_hi == self.sc_lo
                     else self.sc_lo + (self.sc_hi - self.sc_lo) * torch.rand_like(omega))
        wk = {} if w is None else {"w": w}
        t_a = self._warp(t)
        I_t, v = self._interp(z, x, t_a)
        aux = {}
        tangent = v
        if guided:
            tangent, g_aux = self._guided_tangent(reader, I_t, v, t, t_a, labels, x, w)
            aux.update(g_aux)
        if self.student_jvp is not None:
            pred, jvp = self.student_jvp(model, I_t, tangent, labels, omega, **wk)
        else:
            pred, jvp = torch.func.jvp(lambda i: model(i, labels, omega=omega, **wk), (I_t,), (tangent,))
        target = pred.detach() + jvp.detach()
        transport = adaptive_mean(pred, target, self.tp, self.tc)
        if self.guide and self.g_diag_frac > 0:   # drawn whenever the guide is configured, as the reference draws it
            diag = torch.rand_like(omega) < self.g_diag_frac
            omega_prime = torch.where(diag, omega, omega_prime)
        I_end, _ = self._interp(z, x, self._warp(omega_prime))
        end_target = x
        if guided and self.g_endpoint:
            end_target, e_aux = self._guided_endpoint(reader, I_end, x, omega_prime, labels, w)
            aux.update(e_aux)
        e_sw = self._late_weight(omega_prime) if step >= self.late_start else None
        if self.student_fwd is not None:
            out = self.student_fwd(model, I_end, labels, omega, return_repa=self.repa, **wk)
            end_pred, proj = out if self.repa else (out, None)
        elif self.repa:
            end_pred, proj = model(I_end, labels, omega=omega, return_repa=True, **wk)
        else:
            end_pred = model(I_end, labels, omega=omega, **wk)
        if w is not None and guided:
            aux["guide_scale_mean"] = w.mean().detach()
        endpoint = adaptive_mean(end_pred, end_target, self.ep, self.ec, sample_weight=e_sw)
        if e_sw is not None:
            aux["endpoint_late_weight_mean"] = e_sw.mean().detach()
        w_t, w_e, b_aux = self._balance(model, transport, endpoint, step)
        losses = {"transport": w_t * transport, "endpoint": w_e * endpoint}
        if self.transport_weight != 1.0:
            losses["transport"] = self.transport_weight * losses["transport"]
        if self.endpoint_weight != 1.0:
            losses["endpoint"] = self.endpoint_weight * losses["endpoint"]
        if self.repa:
            repa, r_aux = self._repa_loss(proj, x)
            losses["repa"] = repa
            aux.update(r_aux)
        aux.update(b_aux, transport_unweighted=transport.detach(), endpoint_unweighted=endpoint.detach(),
                   interpolant_t_mean=t_a.mean().detach(), nominal_t_mean=t.mean().detach())
        return losses, aux

    def _repa_loss(self, proj, x):
        """REPA on the clean latent: 1 - cos(projection of the trunk's block-k tokens, the clean latent's tokens), mean
        over tokens and the batch, times weight * (elements per example). The endpoint term sums its squared error over
        every latent element, so the per-element factor puts `weight` on the scale it has against a per-element-mean
        flow-matching loss (REPA's and RAEv2's convention)."""
        target = x.flatten(2).transpose(1, 2)                # (B, H*W, C): PatchEmbed's token order at patch 1
        if proj.shape != target.shape:
            raise ValueError(f"guided_cbm.repa: projection {tuple(proj.shape)} vs clean-latent tokens {tuple(target.shape)} "
                             "(set model.repa_z_dim to the latent's channels)")
        cos = torch.nn.functional.cosine_similarity(proj.float(), target.float().detach(), dim=-1)
        per_elem = float(x[0].numel())
        return self.repa_weight * per_elem * (1.0 - cos).mean(), {"repa_cosine": cos.mean().detach()}

    def _guide_weights(self, w, like):
        """(kappa, autoguide weight): the configured numbers, or per example from the scale w (scale_cond):
        (w - 1) / (w + a) and a / (w + a), shaped to broadcast against `like`."""
        if w is None:
            return self.g_kappa, self.ag_weight
        kappa = (w - 1.0) / (w + self.sc_a)
        lam = self.sc_a / (w + self.sc_a)
        return _bcast(kappa, like).to(like.dtype), _bcast(lam, like).to(like.dtype)

    def _guided_tangent(self, reader, I_t, v, t, t_a, labels, x, w=None):
        """V + kappa (b(y) - b(null)) inside [t_lo, t_hi] (nominal) for non-null labels, V elsewhere (the reference w = 1 form)."""
        inside = (t >= self.g_tlo) & (t <= self.g_thi) & ~self._null_mask(labels)
        kappa, agw = self._guide_weights(w, v)
        wk = {} if w is None else {"w": w}
        with torch.no_grad():
            neg = self._null_labels(labels)
            s = _bcast(1.0 - t_a, I_t)
            b_neg = (reader(I_t, neg, omega=t, **wk) - I_t) / s
            guided = v + (1.0 - 1.0 - kappa) * b_neg   # the reference w V + (1 - w - kappa) b_neg at w = 1
            den = reader(I_t, labels, omega=t, **wk)
            b_y = (den - I_t) / s
            guided = guided + kappa * b_y               # + kappa b_y (v_weight 1: no swap of V)
            base = v
            if self.g_marginal:
                # iMF's marginal tangent: b(y) replaces V where the denoiser read at anchor t is trained, and the
                # guide adds kappa (b(y) - b(null)) on top of it; V elsewhere
                marg = (t >= self.m_tlo) & (t <= self.m_thi)
                base = torch.where(_bcast(marg, v), b_y.to(v.dtype), v)
                guided = base + (kappa * (b_y - b_neg)).to(v.dtype)
            if self.ag:   # autoguidance: + weight (b(y) - b_weak(y)), the weak model read at the same (I_t, y, anchor t)
                if self.weak is None:
                    raise RuntimeError("loss.guided_cbm.guide.autoguide: the trainer has not set the weak model (loss.weak)")
                b_w = (self.weak(I_t, labels, omega=t, **wk) - I_t) / s
                ag_shift = (agw * (b_y - b_w)).to(v.dtype)
                guided = guided + ag_shift
        # no_grad forwards inside autocast leave detached cast weights in the cache (a note in the reference code); clear it
        torch.clear_autocast_cache()
        tangent = torch.where(_bcast(inside, v), guided.to(v.dtype), base)
        with torch.no_grad():
            m = inside.float()
            n = m.sum().clamp_min(1.0)
            shift_sq = (tangent - v).float().flatten(1).pow(2).sum(1)
            err = (den - x).float().flatten(1).pow(2).mean(1)
            aux = {"guide_frac": m.mean(), "guide_shift_rms": (shift_sq * m).sum().div(n).sqrt(),
                   "guide_denoiser_mse": (err * m).sum().div(n)}
            if self.ag:   # how far the autoguidance term alone moves the tangent on the guided band
                ag_sq = ag_shift.float().flatten(1).pow(2).sum(1)
                aux["autoguide_shift_rms"] = (ag_sq * m).sum().div(n).sqrt()
            if self.g_marginal:   # how often, and how far, the marginal tangent moves off V
                mm = marg.float()
                d_sq = (base - v).float().flatten(1).pow(2).sum(1)
                aux.update(marginal_frac=mm.mean(), marginal_shift_rms=(d_sq * mm).sum().div(mm.sum().clamp_min(1.0)).sqrt())
        return tangent, aux

    def _guided_endpoint(self, reader, I_end, x, omega_prime, labels, w=None):
        """x + kappa (D(y) - D(null)) for omega' in [t_lo, t_hi] (nominal) and non-null labels, x elsewhere."""
        inside = (omega_prime >= self.g_tlo) & (omega_prime <= self.g_thi) & ~self._null_mask(labels)
        kappa, agw = self._guide_weights(w, x)
        wk = {} if w is None else {"w": w}
        with torch.no_grad():
            neg = self._null_labels(labels)
            d_neg = reader(I_end, neg, omega=omega_prime, **wk)
            guided = 1.0 * x + (1.0 - 1.0 - kappa) * d_neg
            d_y = reader(I_end, labels, omega=omega_prime, **wk)
            guided = guided + kappa * d_y
            if self.ag:   # autoguidance: + weight (D(y) - D_weak(y))
                guided = guided + agw * (d_y - self.weak(I_end, labels, omega=omega_prime, **wk))
        torch.clear_autocast_cache()
        target = torch.where(_bcast(inside, x), guided.to(x.dtype), x)
        with torch.no_grad():
            m = inside.float()
            sq = (target - x).float().flatten(1).pow(2).sum(1)
            aux = {"guide_endpoint_frac": m.mean(),
                   "guide_endpoint_shift_rms": (sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()}
        return target, aux

    def _late_weight(self, omega_prime):
        """min{((1 - t0) / (1 - omega'))^power, cap} above t0 (nominal), 1 below; None when off."""
        if self.late_weight is None:
            return None
        t0, power, cap = self.late_weight
        op = omega_prime.float()
        w = ((1.0 - t0) / (1.0 - op).clamp_min(1e-6)).pow(power).clamp(max=cap)
        return torch.where(op > t0, w, torch.ones_like(w)).detach()

    def _balance(self, model, transport, endpoint, step):
        one = torch.ones((), device=transport.device)
        if self.balance == "none":
            return one, one, {}
        if not torch.is_grad_enabled():   # validation: no gradients to balance; the frozen weight (or 1) for the log
            w = self._frozen_weight if self._frozen_weight is not None and step >= self.g_start else 1.0
            return torch.full((), w, device=transport.device), one, {}
        if step >= self.g_start:   # frozen from the guide start on
            if self._frozen_weight is None:
                if self._ratio_ema is None:
                    raise RuntimeError("loss.guided_cbm.balance.mode freeze reached guide.start_step with no balancer history "
                                       "(a resume past the freeze needs the checkpoint's loss state)")
                w = self._ratio_ema.to(device=transport.device, dtype=torch.float64).clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(w)
                    w /= dist.get_world_size()
                self._frozen_weight = float(w)
                print(f"[guided_cbm] step {step}: transport weight frozen at {self._frozen_weight:.6g}", flush=True)
            w_t = torch.full((), self._frozen_weight, device=transport.device)
            return w_t, one, {"balance_transport_weight": w_t}
        module = model
        for part in self.balance_path.split("."):
            module = getattr(module, part)
        params = tuple(p for p in module.parameters() if p.requires_grad)
        n_e, n_t = _grad_norm(endpoint, params), _grad_norm(transport, params)
        w_e = torch.where(n_e < n_t, torch.clamp_min(n_t / (n_e + 1e-8), 1.0), one).detach()
        w_t = torch.where(n_t < n_e, torch.clamp_min(n_e / (n_t + 1e-8), 1.0), one).detach()
        # the EMA of |g_E| / |g_T| stays on the device (no host sync per step); a non-finite ratio leaves it unchanged
        ratio = (n_e / (n_t + 1e-8)).detach().float()
        if self._ratio_ema is None:
            self._ratio_ema = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))
        else:
            ema = self._ratio_ema.to(ratio.device)
            d = self.balance_ema
            self._ratio_ema = torch.where(torch.isfinite(ratio), d * ema + (1.0 - d) * ratio, ema)
        return w_t, w_e, {"balance_transport_weight": w_t, "balance_endpoint_weight": w_e,
                          "final_layer_grad_norm_transport": n_t.detach(), "final_layer_grad_norm_endpoint": n_e.detach(),
                          "balance_ratio_ema": self._ratio_ema}


def get_guided_cbm_loss_fn(cfg):
    return GuidedCBMLoss(cfg.loss.guided_cbm, label_dim=cfg.dataset.num_classes)
