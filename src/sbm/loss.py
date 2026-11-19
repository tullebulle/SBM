"""The SBM objective: the BTM rule (transport + endpoint) with self-guidance in the targets.

Notation: I_t = (1 - t) z + t x (t = 1 is data), T(.; y, omega) the map, omega the anchor time (network input).

    transport    || T(I_t) - sg(T(I_t) + J_T(I_t) (x - z)) ||^2 * sg[(d^2 + c)^-p]      (the JVP target, adaptive weight)
    endpoint     || T(I_omega') - x ||^2 * sg[(d^2 + c)^-p],   omega' ~ U[omega, 1]      (pins T = id on data)
                 endpoint.omega_skew k > 1 draws omega' = omega + (1 - omega)(1 - (1 - u)^k), closer to 1 (a sharper target);
                 endpoint.late_weight {t0, power, cap}: rows at omega' > t0 weighted min{((1 - t0) / (1 - omega'))^power, cap};
                 endpoint.weight / transport.weight scale each term's own gradient only (the balancer sees the unweighted ones).
    balance      the per-step final-layer balancer (match_smaller_to_larger: the smaller of the two final-layer gradient
                 norms is scaled up to the larger), or (mode ema) the same ratio on EMA-smoothed norms, or none (fixed
                 transport.weight / endpoint.weight, the XL line's frozen ratio 2000 / 1).
    time         logit-normal transport times (p_mean annealed to p_mean_end over anneal_steps from anneal_start);
                 time.mix_frac of them uniform on [mix_lo, omega) (the late path the one-step read flows through).
                 time.importance (off by default): eps of the rows redrawn U(0, 1) and every transport row weighted
                 1 / p_eps(t | omega) - the uniform-time population objective under the recipe's draws (bounded weights).
    guide        (the SBM recipes' self-guidance) iMF-style guidance in the transport
                 target: for transport times t in [t_lo, t_hi] and non-null labels the tangent V = x - z becomes
                 w V + kappa b(I_t | y) + (1 - w - kappa) b(I_t | neg), with b(I_t | c) = (T(I_t; c, omega = t) - I_t) / (1 - t)
                 the map's own denoiser velocity (no gradient; neg = the null class, or a rolled class). In expectation the
                 CFG field b_{w + kappa}, so the fixed point transports along the guided flow. w = 1, kappa = g - 1 is the
                 shift form V + (g - 1)(b_y - b_neg) (the recipes: w 1, kappa 0.8 on B/2, 0.7 on XL). guide.endpoint: the
                 endpoint target inside the interval is w x + kappa D(y) + (1 - w - kappa) D(neg), D(c) = T(I_omega'; c,
                 omega = omega'); guide.diag_frac sets omega' = omega on that fraction of the endpoint rows (iMF's r = t
                 rows). The reads are made by the online map or (loss.guide.teacher_decay) an EMA copy handed in by the
                 trainer (the recipes: the frozen decay-1.0 copy refreshed every 10k steps). Needs the null class trained
                 (label_dropout > 0) and anchors down to t_lo (anchor_alpha >= 1 - t_lo).
    guide.omega_cond  iMF's Omega-conditioning (off by default): per example w ~ w^-beta on [w_min, w_max] and an
                 interval [s_lo, s_hi] fed to a map with model.gw_embedder (gw_input kappa) and model.gw_interval; inside
                 the interval the tangent is V + (1 - 1/w)(b(y, w) - b(null, 1)) and the endpoint target
                 x + (1 - 1/w)(D(y, w) - D(null, 1)).
    guide.autoguide  (off by default; Karras et al. 2024, arXiv 2406.02507) autoguidance on the same guided band and rows: the
                 tangent gains + weight (b(y) - b_weak(y)) and the endpoint target + weight (D(y) - D_weak(y)), b_weak /
                 D_weak the same reads by a frozen WEAK copy of the map (an earlier checkpoint of the run, loaded once by
                 the trainer, never refreshed). The teacher compounds kappa and weight, so kappa + weight < 1. Off (the
                 default) leaves every computation as is.

Engineering switches (configs/sbm/train.yaml; every default leaves the computation unchanged):
    sync_free            the logged scalars are made on the device (no pageable host-to-device copy, which syncs).
    balance.refresh_every  (mode ema) the two final-layer norms are measured every N micro-batches and the cached weights
                         reused between refreshes (the EMA's decay horizon preserved); 1 is the recipe.
    balance.reset_on_resume  a continuation that changes either balanced objective re-bootstraps the norm EMA.

The draws come from the global RNG in a fixed order (z, t, omega, u, [mixture], [dropout], ...), and the balancer runs
on the final layer inside the step; the goldens (tests/sbm/golden) pin this path.
"""
from __future__ import annotations

import math

import torch


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
    """The objective above (the class name and its checkpoint key `cbm_loss_state` are kept so that existing checkpoints
    resume unchanged)."""

    def __init__(self, cfg, label_dim=1000, sync_free=False):
        """cfg: the `loss` block of the run config (configs/sbm/train.yaml)."""
        self.label_dim = int(label_dim)
        self.sync_free = bool(sync_free)
        self.tp, self.tc = float(cfg.transport.p), float(cfg.transport.c)
        self.ep, self.ec = float(cfg.endpoint.p), float(cfg.endpoint.c)
        self.endpoint_weight = float(cfg.endpoint.get("weight", 1.0))
        self.transport_weight = float(cfg.transport.get("weight", 1.0))   # scales the weighted transport term only
        self.omega_skew = float(cfg.endpoint.get("omega_skew", 1.0))
        # endpoint rows at omega' > t0 weighted min{((1 - t0) / (1 - omega'))^power, cap} (the epsilon-parametrisation's weight
        # on the late band, where plain L2 on x leaves a residual-noise gradient ~ (1 - omega')); the XL line: {0.6, 1, 8}
        lw = cfg.endpoint.get("late_weight")
        self.late_weight = None if not lw else (float(lw["t0"]), float(lw.get("power", 2.0)), float(lw.get("cap", 64.0)))
        self.anchor_alpha = float(cfg.anchor_alpha)
        self.restrict = bool(cfg.restrict_transport)
        t = cfg.time
        self.p_mean0, self.p_std, self.eps = float(t.p_mean), float(t.p_std), float(t.eps)
        self.p_mean_end = None if t.get("p_mean_end") is None else float(t.p_mean_end)
        self.anneal_start, self.anneal_steps = int(t.get("anneal_start", 0)), int(t.get("anneal_steps", 0))
        self.p_mean = self.p_mean0
        # a fraction of the transport times uniform on [mix_lo, omega), the late path the one-step read flows through
        # (logit-normal -1.2 puts 5% of its mass on t > 0.6); t stays below the example's anchor, so no transport sample
        # lands in the endpoint band
        self.t_mix, self.t_mix_lo = float(t.get("mix_frac", 0.0)), float(t.get("mix_lo", 0.3))
        # importance weights on the transport-time law (a non-uniform time law with the unweighted direction makes the map
        # conserve J_p = int p rho_t b_t instead of the BTM current): with probability eps a
        # row's transport time is redrawn U(0, 1), and every transport row is weighted 1 / p_eps(t | omega), p_eps = (1 - eps) p + eps
        # with p the recipe's law (1 - mix_frac) logit-normal(p_mean, p_std) + mix_frac U(mix_lo, omega). The population transport
        # objective is then the uniform-time one; the weights are bounded by 1 / eps. Off (the default): no draw, no weight.
        im = t.get("importance") or {}
        self.t_imp, self.t_imp_eps = bool(im.get("enabled", False)), float(im.get("eps", 0.1))
        if self.t_imp and not 0.0 < self.t_imp_eps <= 1.0:
            raise ValueError(f"loss.time.importance.eps must be in (0, 1], got {self.t_imp_eps}")
        if self.t_imp and self.restrict:
            raise ValueError("loss.time.importance is written for the time law on (0, 1): restrict_transport must be false")
        self.balance = str(cfg.balance.mode)
        self.balance_path = str(cfg.balance.get("parameter_path", "final_layer"))
        # mode ema: the two final-layer norms are smoothed with this decay before the ratio is taken, so the transport
        # weight (~2000 under match_smaller_to_larger) no longer jitters x6 from step to step with the per-batch norms
        self.balance_ema = float(cfg.balance.get("ema", 0.99))
        self.balance_refresh_every = int(cfg.balance.get("refresh_every", 1))
        if self.balance_refresh_every < 1 or (self.balance_refresh_every > 1 and self.balance != "ema"):
            raise ValueError("loss.balance.refresh_every > 1 requires mode=ema; interval must be positive")
        self._balance_calls, self._balance_cache = 0, None
        # A continuation that changes either balanced objective must retain model/optimizer/EMA state but re-bootstrap
        # the norm EMA from the new pair. Off by default, so ordinary resumes remain bit-for-bit.
        self.balance_reset_on_resume = bool(cfg.balance.get("reset_on_resume", False))
        self._balance_norms = None
        self.label_dropout = float(cfg.get("label_dropout", 0.0))
        gd = cfg.get("guide") or {}
        self.guide = bool(gd.get("enabled", False))
        self.g_w, self.g_kappa = float(gd.get("w", 1.0)), float(gd.get("kappa", 0.0))
        self.g_tlo, self.g_thi = float(gd.get("t_lo", 0.35)), float(gd.get("t_hi", 0.6))
        self.g_negative, self.g_start = str(gd.get("negative", "uncond")), int(gd.get("start_step", 0))
        # iMF's guided boundary: inside the interval the endpoint target is w x + kappa D(I_w' | y) + (1 - w - kappa) D(I_w' | neg),
        # D the map's own diagonal read T(I_w'; c, omega = w'); diag_frac sets omega' = omega on that fraction of the endpoint
        # rows (iMF's r = t rows)
        self.g_endpoint, self.g_diag_frac = bool(gd.get("endpoint", False)), float(gd.get("diag_frac", 0.0))
        # iMF's Omega-conditioning: per example w ~ w^-beta on [w_min, w_max] and an interval s_lo ~ U[0, 0.5], s_hi ~ U[0.5, 1]
        # (diagonal rows [0, 1]), fed to the map (model.gw_embedder with gw_input kappa, model.gw_interval); inside the interval
        # the tangent is V + (1 - 1/w)(b(y, w) - b(null, 1)) and the endpoint target x + (1 - 1/w)(D(y, w) - D(null, 1)), the
        # reads at omega = t (omega') with the interval [0, 1], no gradient
        self.g_omega_cond = bool(gd.get("omega_cond", False))
        self.g_wmin, self.g_wmax, self.g_beta = float(gd.get("w_min", 1.0)), float(gd.get("w_max", 8.0)), float(gd.get("beta", 1.0))
        # t_cap: the guided tangent only at t <= t_cap. The velocity read is a denoiser displacement over (1 - t), so near data
        # the read's error is amplified by 1 / (1 - t). 1.0 = no cap. The guided endpoint has no division: not capped.
        self.g_tcap = float(gd.get("t_cap", 1.0))
        # the sample velocity x - z in the guided tangent is replaced by the (teacher's) conditional velocity read with weight
        # 1 - v_weight; 1 = the recipe, 0 = iMF's predicted tangent b_y + kappa (b_y - b_neg); inside the guided interval only
        self.g_vw = float(gd.get("v_weight", 1.0))
        if self.guide and self.g_omega_cond:
            if self.g_negative != "uncond" or not self.g_endpoint:
                raise ValueError("loss.guide.omega_cond is iMF's recipe: negative uncond and the guided endpoint (endpoint true)")
            if not 1.0 <= self.g_wmin <= self.g_wmax:
                raise ValueError("loss.guide.omega_cond needs 1 <= w_min <= w_max")
        if self.guide:
            if self.g_negative not in ("uncond", "rolled"):
                raise ValueError(f"loss.guide.negative must be uncond or rolled, got {self.g_negative}")
            if self.g_negative == "uncond" and self.label_dropout <= 0:
                raise ValueError("loss.guide.negative=uncond needs the null class trained (loss.label_dropout > 0)")
            if not 0.0 < self.g_tlo <= self.g_thi < 1.0 or 1.0 - self.anchor_alpha > self.g_tlo + 1e-9:
                raise ValueError(f"loss.guide interval [{self.g_tlo}, {self.g_thi}] must lie in (0, 1) and the anchors must reach "
                                 f"its start (1 - anchor_alpha = {1.0 - self.anchor_alpha:.3f} <= t_lo): the denoiser T(I_t; omega=t)")
        # autoguidance (module docstring `guide.autoguide`): the weak model itself is built and frozen by the trainer
        # (sbm.train.Module._guide_weak, from autoguide.ckpt / which) and handed to __call__ as `weak`; the loss only reads it
        ag = gd.get("autoguide") or {}
        self.ag = bool(ag.get("enabled", False))
        self.ag_weight = float(ag.get("weight", 0.0) or 0.0)
        self.ag_ckpt, self.ag_which = ag.get("ckpt"), str(ag.get("which", "ema"))
        if self.ag:
            if not self.guide or not self.ag_ckpt:
                raise ValueError("loss.guide.autoguide needs the guide on (loss.guide.enabled) and a ckpt (the weak model's checkpoint)")
            if self.ag_weight < 0:
                raise ValueError(f"loss.guide.autoguide.weight must be >= 0, got {self.ag_weight}")
            if self.g_kappa + self.ag_weight >= 1.0:
                raise ValueError(f"loss.guide.autoguide needs kappa + weight < 1 (the self-referential teacher compounds both, so "
                                 f"the guidance has no finite fixed point otherwise), got {self.g_kappa} + {self.ag_weight}")
            if self.g_omega_cond:
                raise ValueError("loss.guide.autoguide is implemented for the fixed-kappa guide only (no omega_cond)")

    # ------------------------------------------------------------------ draws
    def _time(self, x):
        t = torch.sigmoid(torch.randn(x.shape[0], device=x.device, dtype=x.dtype) * self.p_std + self.p_mean)
        return t * (1 - 2 * self.eps) + self.eps if self.eps > 0 else t

    def _omega(self, x):
        if self.anchor_alpha == 0.0:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        return 1.0 - self.anchor_alpha + self.anchor_alpha * torch.rand(x.shape[0], device=x.device, dtype=x.dtype)

    @staticmethod
    def _interp(z, x, s):
        """(I_s, Idot_s) with alpha_s = 1 - s (the operation order is fixed: the goldens pin it)."""
        a = _bcast(1.0 - s, x)
        return a * z + (1.0 - a) * x, -1.0 * (z - x)

    def _anneal(self, step):
        if self.p_mean_end is not None and self.anneal_steps > 0:
            frac = min(1.0, max(0.0, (float(step or 0) - self.anneal_start) / self.anneal_steps))
            self.p_mean = self.p_mean0 + frac * (self.p_mean_end - self.p_mean0)

    # ------------------------------------------------------------------ the step
    def __call__(self, model, x, labels, step, teacher=None, weak=None):
        """-> (weighted terms to be summed and backpropagated, aux scalars to log). `teacher` (loss.guide.teacher_decay):
        a slower copy (an EMA) that makes the fixed-kappa guide's reads instead of the online map. `weak` (loss.guide.autoguide):
        the frozen weak model whose reads the autoguidance term subtracts; required when autoguide is on."""
        labels = labels.long()
        self._anneal(step)
        z = torch.randn_like(x)
        t = self._time(x)
        omega = self._omega(x)
        if self.restrict:            # transport time inside [0, omega], where the endpoint term does not pin T
            t = t * omega
        if self.t_mix > 0:           # drawn only when on, so the default draws are unchanged
            late = torch.rand_like(t) < self.t_mix
            t = torch.where(late, self.t_mix_lo + (omega - self.t_mix_lo).clamp_min(0.0) * torch.rand_like(t), t)
        t_sw = None
        if self.t_imp:               # drawn only when on, so the default draws are unchanged
            unif = torch.rand_like(t) < self.t_imp_eps
            t = torch.where(unif, torch.rand_like(t), t)
            t_sw = self._importance_weight(t, omega)
        u = torch.rand_like(omega)
        if self.omega_skew != 1.0:   # the anchor's target sharpens as omega' -> 1 (less of a conditional mean)
            u = 1.0 - (1.0 - u) ** self.omega_skew
        omega_prime = omega + (1.0 - omega) * u
        if self.label_dropout > 0:   # label dropout to the null class (the recipes: 0.1; the guide reads the null class)
            drop = torch.rand(labels.shape, device=labels.device) < self.label_dropout
            labels = torch.where(drop, torch.full_like(labels, self.label_dim), labels)
        g_w = g_lo = g_hi = None
        if self.guide and self.g_omega_cond and int(step or 0) >= self.g_start:
            g_w, g_lo, g_hi = self._draw_omega(x)
        ikw = {} if g_lo is None else {"s_lo": g_lo, "s_hi": g_hi}
        I_t, v = self._interp(z, x, t)
        aux = {} if g_w is None else {"guide_w_mean": g_w.float().mean().detach()}
        tangent, g_inside = v, None
        if g_w is not None:
            tangent, g_aux, g_inside = self._guided_tangent_omega(model, I_t, v, t, labels, g_w, g_lo, g_hi)
            aux.update(g_aux)
        elif self.guide and int(step or 0) >= self.g_start:
            tangent, g_aux, g_inside = self._guided_tangent(model if teacher is None else teacher, I_t, v, t, labels, x, weak=weak)
            aux.update(g_aux)
        pred, jvp = torch.func.jvp(lambda i: model(i, labels, omega=omega, w=g_w, **ikw), (I_t,), (tangent,))
        target = pred.detach() + jvp.detach()
        if self.guide and "guide_frac" in aux:   # the transport weight's spread where the guided tangent acts (a
            with torch.no_grad():                # concentrated weight starves the guidance of traction)
                wt = ((pred - target).float().flatten(1).pow(2).sum(1) + self.tc) ** -self.tp
                m = g_inside.float()
                n = m.sum().clamp_min(1.0)
                mean = (wt * m).sum() / n
                aux["guide_weight_cv"] = (((wt - mean).pow(2) * m).sum() / n).sqrt() / mean.clamp_min(1e-12)
        transport = adaptive_mean(pred, target, self.tp, self.tc, sample_weight=t_sw)
        if t_sw is not None:
            aux["transport_importance_mean"], aux["transport_importance_max"] = t_sw.mean().detach(), t_sw.max().detach()
        diag = None
        if self.guide and self.g_diag_frac > 0:   # drawn only when on, so the default draws are unchanged
            diag = torch.rand_like(omega) < self.g_diag_frac
            omega_prime = torch.where(diag, omega, omega_prime)
        I_end, _ = self._interp(z, x, omega_prime)
        end_target, e_kw = x, ikw
        if g_w is not None:   # diagonal rows are always guided and see the interval [0, 1] (iMF's r = t rows)
            e_lo, e_hi = g_lo, g_hi
            if diag is not None:
                e_lo, e_hi = torch.where(diag, torch.zeros_like(g_lo), g_lo), torch.where(diag, torch.ones_like(g_hi), g_hi)
            e_kw = {"s_lo": e_lo, "s_hi": e_hi}
            end_target, e_aux = self._guided_endpoint_omega(model, I_end, x, omega_prime, labels, g_w, e_lo, e_hi)
            aux.update(e_aux)
        elif self.guide and self.g_endpoint and int(step or 0) >= self.g_start:
            end_target, e_aux = self._guided_endpoint(model if teacher is None else teacher, I_end, x, omega_prime, labels, weak=weak)
            aux.update(e_aux)
        e_sw = self._late_weight(omega_prime)
        endpoint = adaptive_mean(model(I_end, labels, omega=omega, w=g_w, **e_kw), end_target, self.ep, self.ec, sample_weight=e_sw)
        if e_sw is not None:
            aux["endpoint_late_weight_mean"] = e_sw.mean().detach()
        w_t, w_e, bal_aux = self._balance(model, transport, endpoint)
        losses = {"transport": w_t * transport, "endpoint": w_e * endpoint}
        if self.transport_weight != 1.0:
            losses["transport"] = self.transport_weight * losses["transport"]
        if self.endpoint_weight != 1.0:
            losses["endpoint"] = self.endpoint_weight * losses["endpoint"]
        aux.update(bal_aux, transport_unweighted=transport.detach(), endpoint_unweighted=endpoint.detach(),
                   interpolant_t_mean=t.mean().detach(), time_p_mean=self._scalar(self.p_mean, x.device))
        return losses, aux

    def _guided_tangent(self, model, I_t, v, t, labels, x=None, weak=None):
        """w V + kappa b(I_t | y) + (1 - w - kappa) b(I_t | neg) inside the interval (non-null labels), V elsewhere; the
        velocities are the map's own denoiser read at omega = t, without gradient. Autoguidance (loss.guide.autoguide) adds
        + weight (b(I_t | y) - b_weak(I_t | y)) on the same rows, b_weak the frozen weak model's read at the same (I_t, y, omega = t)."""
        ag = self.ag
        if ag and weak is None:
            raise RuntimeError("loss.guide.autoguide is on but no weak model was passed (the trainer loads it from autoguide.ckpt)")
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
            if self.g_kappa or self.g_vw != 1.0 or ag:
                den = model(I_t, labels, omega=t)
                b_y = (den - I_t) / s
                # kappa b_y, and (1 - v_weight) of the sample velocity swapped for the conditional read (v_weight 1: none)
                guided = guided + self.g_kappa * b_y + (1.0 - self.g_vw) * (b_y - v)
            if ag:   # autoguidance: + weight (b(y) - b_weak(y)), the weak model read at the same (I_t, y, omega = t) as the teacher
                ag_shift = self.ag_weight * (b_y - (weak(I_t, labels, omega=t) - I_t) / s)
                guided = guided + ag_shift
        # a no_grad forward inside autocast leaves detached cast weights in the cache; the grad forwards after it would
        # then reach only the fp32 parameters (measured: the matrices' update ~10x too small). Clear it.
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
            if ag:   # how far the autoguidance term alone moves the tangent on the guided band
                ag_sq = ag_shift.float().flatten(1).pow(2).sum(1)
                aux["autoguide_shift_rms"] = (ag_sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()
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

    def _time_density(self, t, omega):
        """p(t | omega) of the recipe's transport-time law, a density w.r.t. U(0, 1): (1 - mix_frac) logit-normal(p_mean, p_std) +
        mix_frac U(mix_lo, omega) (the current, possibly annealed, p_mean; the 1e-4 clamp of _time is ignored)."""
        tf = t.float().clamp(1e-6, 1.0 - 1e-6)
        z = (torch.log(tf / (1.0 - tf)) - self.p_mean) / self.p_std
        ln = torch.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * self.p_std * tf * (1.0 - tf))
        if self.t_mix <= 0:
            return ln
        om = omega.float()
        un = ((tf >= self.t_mix_lo) & (tf < om)).float() / (om - self.t_mix_lo).clamp_min(1e-6)
        return (1.0 - self.t_mix) * ln + self.t_mix * un

    def _importance_weight(self, t, omega):
        """loss.time.importance: 1 / p_eps(t | omega), p_eps = (1 - eps) p + eps; bounded by 1 / eps, mean 1 under the p_eps draw."""
        p_eps = (1.0 - self.t_imp_eps) * self._time_density(t, omega) + self.t_imp_eps
        return (1.0 / p_eps).detach()

    def _late_weight(self, omega_prime):
        """loss.endpoint.late_weight: per-row weight min{((1 - t0) / (1 - omega'))^power, cap} above t0, 1 below; None when off."""
        if self.late_weight is None:
            return None
        t0, power, cap = self.late_weight
        op = omega_prime.float()
        w = ((1.0 - t0) / (1.0 - op).clamp_min(1e-6)).pow(power).clamp(max=cap)
        return torch.where(op > t0, w, torch.ones_like(w)).detach()

    def _guided_endpoint(self, model, I_end, x, omega_prime, labels, weak=None):
        """The endpoint target with guidance (iMF's guided boundary): w x + kappa D(y) + (1 - w - kappa) D(neg) for
        omega' in [t_lo, t_hi] and non-null labels, x elsewhere; D(c) = T(I_omega'; c, omega = omega'), no gradient.
        Autoguidance adds + weight (D(y) - D_weak(y)) on the same rows, D_weak the frozen weak model's read at (I_omega', y, omega')."""
        ag = self.ag
        if ag and weak is None:
            raise RuntimeError("loss.guide.autoguide is on but no weak model was passed (the trainer loads it from autoguide.ckpt)")
        inside = (omega_prime >= self.g_tlo) & (omega_prime <= self.g_thi) & (labels != self.label_dim)
        with torch.no_grad():
            if self.g_negative == "uncond":
                neg = torch.full_like(labels, self.label_dim)
            else:
                shift = 1 + int(torch.randint(labels.shape[0] - 1, (1,), device="cpu"))
                neg = torch.roll(labels, shift)
            d_neg = model(I_end, neg, omega=omega_prime)
            guided = self.g_w * x + (1.0 - self.g_w - self.g_kappa) * d_neg
            d_y = model(I_end, labels, omega=omega_prime) if (self.g_kappa or ag) else None
            if self.g_kappa:
                guided = guided + self.g_kappa * d_y
            if ag:   # autoguidance: + weight (D(y) - D_weak(y)), the weak model read at the same (I_omega', y, omega = omega')
                guided = guided + self.ag_weight * (d_y - weak(I_end, labels, omega=omega_prime))
        torch.clear_autocast_cache()
        target = torch.where(_bcast(inside, x), guided.to(x.dtype), x)
        with torch.no_grad():
            m = inside.float()
            sq = (target - x).float().flatten(1).pow(2).sum(1)
            aux = {"guide_endpoint_frac": m.mean(),
                   "guide_endpoint_shift_rms": (sq * m).sum().div(m.sum().clamp_min(1.0)).sqrt()}
        return target, aux

    def _scalar(self, value, device):
        if self.sync_free:
            return torch.full((), float(value), device=device)
        return torch.as_tensor(float(value), device=device)

    # ------------------------------------------------------------------ state that rides in the checkpoint
    def state_dict(self):
        out = {}
        if self._balance_norms is not None:
            out["balance_norms"] = tuple(n.detach().cpu() for n in self._balance_norms)
        if self.balance_refresh_every > 1 and self._balance_cache is not None:
            out["balance_cache"] = {"every": self.balance_refresh_every, "calls": self._balance_calls,
                                    "values": tuple(n.detach().cpu() for n in self._balance_cache)}
        return out

    def load_state_dict(self, state):
        if self.balance == "ema" and not self.balance_reset_on_resume and "balance_norms" in state:
            self._balance_norms = tuple(torch.as_tensor(n) for n in state["balance_norms"])
            cache = state.get("balance_cache")
            if self.balance_refresh_every > 1 and cache and int(cache["every"]) == self.balance_refresh_every:
                self._balance_calls = int(cache["calls"])
                self._balance_cache = tuple(torch.as_tensor(n) for n in cache["values"])

    def _balance(self, model, transport, endpoint):
        """match_smaller_to_larger on the final layer: the smaller of the two gradient norms is scaled up to the larger
        (mode ema: on the EMA-smoothed norms; none: both weights 1)."""
        one = torch.ones((), device=transport.device)
        if self.balance == "none":
            return one, one, {}
        cached_mode = self.balance_refresh_every > 1
        had_cache = self._balance_cache is not None
        if cached_mode:
            refresh = not had_cache or self._balance_calls % self.balance_refresh_every == 0
            self._balance_calls += 1
            if not refresh:
                self._balance_cache = tuple(n.to(transport.device) for n in self._balance_cache)
                w_t, w_e, raw_t, raw_e = self._balance_cache
                return w_t, w_e, {"balance_transport_weight": w_t, "balance_endpoint_weight": w_e,
                                  "final_layer_grad_norm_transport": raw_t, "final_layer_grad_norm_endpoint": raw_e,
                                  "balance_refreshed": one * 0}
        module = model
        for part in self.balance_path.split("."):
            module = getattr(module, part)
        params = tuple(p for p in module.parameters() if p.requires_grad)
        n_e, n_t = grad_norm(endpoint, params), grad_norm(transport, params)
        raw_e, raw_t = n_e.detach(), n_t.detach()
        if self.balance == "ema":   # the smoothed norms set the weights; the raw ones are still logged
            if self._balance_norms is None:
                self._balance_norms = (raw_e.clone(), raw_t.clone())
            else:
                d, (m_e, m_t) = self.balance_ema, (n.to(raw_e.device) for n in self._balance_norms)
                if cached_mode and had_cache:
                    # Preserve the EMA's elapsed-microbatch decay when a fresh observation replaces the skipped ones.
                    # This is an approximation, not equivalent to observing every batch.
                    d = d ** self.balance_refresh_every
                self._balance_norms = (d * m_e + (1 - d) * raw_e, d * m_t + (1 - d) * raw_t)
            n_e, n_t = self._balance_norms
        w_e = torch.where(n_e < n_t, torch.clamp_min(n_t / (n_e + 1e-8), 1.0), one).detach()
        w_t = torch.where(n_t < n_e, torch.clamp_min(n_e / (n_t + 1e-8), 1.0), one).detach()
        aux = {"balance_transport_weight": w_t, "balance_endpoint_weight": w_e,
               "final_layer_grad_norm_transport": raw_t, "final_layer_grad_norm_endpoint": raw_e}
        if cached_mode:
            self._balance_cache = (w_t, w_e, raw_t, raw_e)
            aux["balance_refreshed"] = one
        return w_t, w_e, aux
