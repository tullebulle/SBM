"""Loss factory of the sbm_rae package (the ImageNet RAE-latent models and the one-step world models).

Two objectives: loss.name=guided_cbm (sbm_rae.guided_cbm, the self-guided objective of the ImageNet models) and
loss.name=bridge_cbm (sbm_rae.bridge_loss, the same objective on the world model's condition dict). The `anchor`,
`guidance` and `signed` config blocks are still read by the resolvers below, which the trainer, the FID scorer and
scripts/rae/train.py use to size the network (guidance_enabled) and the sampling grid; guidance and signed stay off.
"""
import math


def _canonical_guidance_mode(mode):
	mode = str(mode)
	aliases = {
		"cfg": "cfg",
		"classifier_free": "cfg",
		"negative": "negative",
		"negative_cfg": "negative",
		"negative_guidance": "negative",
	}
	if mode not in aliases:
		raise ValueError("guidance.mode must be one of: cfg, negative")
	return aliases[mode]




def resolve_anchor_alpha(cfg):
	anchor_cfg = cfg.get("anchor", None)

	if anchor_cfg is None:
		alpha = 0.0
	elif isinstance(anchor_cfg, bool):
		alpha = 0.0
	elif isinstance(anchor_cfg, (int, float)):
		alpha = float(anchor_cfg)
	elif isinstance(anchor_cfg, str):
		if anchor_cfg.lower() in {"", "false", "none", "null", "off"}:
			alpha = 0.0
		else:
			alpha = float(anchor_cfg)
	else:
		if not bool(anchor_cfg.get("enabled", True)):
			alpha = 0.0
		else:
			alpha = float(anchor_cfg.get("alpha", 0.0))

	if not 0.0 <= alpha <= 1.0:
		raise ValueError("anchor.alpha must be in [0, 1]")
	return alpha


def resolve_guidance_config(cfg):
	guidance_cfg = cfg.get("guidance", None)
	default_alpha_range = (1.0, 5.0)
	if guidance_cfg is None:
		return {
			"enabled": False,
			"mode": "cfg",
			"w_min": 1.0,
			"class_drop_prob": 0.0,
			"beta": 1.0,
			"alpha_range": default_alpha_range,
		}

	if isinstance(guidance_cfg, bool):
		resolved = {
			"enabled": bool(guidance_cfg),
			"mode": "cfg",
			"w_min": 0.5,
			"class_drop_prob": 0.1,
			"beta": 1.0,
			"alpha_range": default_alpha_range,
		}
	else:
		alpha_range = guidance_cfg.get("alpha_range", default_alpha_range)
		resolved = {
			"enabled": bool(guidance_cfg.get("enabled", True)),
			"mode": _canonical_guidance_mode(guidance_cfg.get("mode", "cfg")),
			"w_min": float(guidance_cfg.get("w_min", 0.5)),
			"class_drop_prob": float(guidance_cfg.get("class_drop_prob", 0.1)),
			"beta": float(guidance_cfg.get("beta", 1.0)),
			"alpha_range": tuple(float(alpha) for alpha in alpha_range),
		}

	if not 0.0 <= resolved["w_min"] <= 1.0:
		raise ValueError("guidance.w_min must be in [0, 1]")
	if not 0.0 <= resolved["class_drop_prob"] <= 1.0:
		raise ValueError("guidance.class_drop_prob must be in [0, 1]")
	if resolved["beta"] <= 0.0:
		raise ValueError("guidance.beta must be > 0")
	if len(resolved["alpha_range"]) != 2:
		raise ValueError("guidance.alpha_range must be a (min, max) pair")
	if not 0.0 < resolved["alpha_range"][0] <= resolved["alpha_range"][1]:
		raise ValueError("guidance.alpha_range must satisfy 0 < min <= max")
	return resolved


def resolve_signed_schedule(schedule_cfg, alpha):
	"""Normalise a `signed.schedule` block; None when absent or disabled.

	The strength at global step s is

	    alpha(s) = warm(s) * base(s)
	    warm(s)  = min(1, s / warmup_steps)              (1 when warmup_steps is 0)
	    base(s)  = alpha_start                           for s <= decay_start_step
	             = linear from alpha_start to alpha_end  on (decay_start_step, decay_end_step)
	             = alpha_end                             for s >= decay_end_step
	             = alpha_start everywhere when there is no decay window (end <= start)

	so the default (no decay, no warm-up) is the constant `alpha`.
	"""
	if not schedule_cfg:
		return None
	if not hasattr(schedule_cfg, "get"):
		raise ValueError("signed.schedule must be a mapping")
	resolved_keys = {"warmup_steps", "alpha_start", "alpha_end", "decay_start_step", "decay_end_step"}
	if set(schedule_cfg.keys()) == resolved_keys:
		# Already resolved (the loss factory hands the resolver's output back in).
		return dict(schedule_cfg)
	if not bool(schedule_cfg.get("enabled", False)):
		return None
	start = schedule_cfg.get("alpha_start", None)
	end = schedule_cfg.get("alpha_end", None)
	resolved = {
		"warmup_steps": int(schedule_cfg.get("warmup_steps", 0) or 0),
		"alpha_start": float(alpha if start is None else start),
		"alpha_end": float(alpha if end is None else end),
		"decay_start_step": int(schedule_cfg.get("decay_start_step", 0) or 0),
		"decay_end_step": int(schedule_cfg.get("decay_end_step", 0) or 0),
	}
	for key in ("alpha_start", "alpha_end"):
		if resolved[key] < 0.0 or not math.isfinite(resolved[key]):
			raise ValueError(f"signed.schedule.{key} must be finite and >= 0")
	if resolved["warmup_steps"] < 0:
		raise ValueError("signed.schedule.warmup_steps must be >= 0")
	if resolved["decay_end_step"] < resolved["decay_start_step"]:
		raise ValueError("signed.schedule.decay_end_step must be >= decay_start_step")
	if resolved["alpha_start"] == 0.0 and resolved["alpha_end"] == 0.0:
		raise ValueError("signed.schedule needs a positive alpha_start or alpha_end")
	return resolved


def resolve_signed_config(cfg):
	"""Read the `signed` block off a training config; absent means the native rule.

	`alpha` is the signed strength, NOT the anchor alpha in the `anchor` block and NOT
	`guidance.alpha_range`. The signed rule itself is not included; the block must stay off.
	"""
	signed_cfg = cfg.get("signed", None) or {}
	if isinstance(signed_cfg, bool):
		signed_cfg = {"enabled": bool(signed_cfg), "alpha": 1.0}
	resolved = {
		"enabled": bool(signed_cfg.get("enabled", False)),
		"alpha": float(signed_cfg.get("alpha", 0.0)),
		"endpoint": bool(signed_cfg.get("endpoint", False)),
		"paired": bool(signed_cfg.get("paired", False)),
		"negative_t_max": float(signed_cfg.get("negative_t_max", 1.0)),
		"project_negative": bool(signed_cfg.get("project_negative", False)),
		"negative_source": str(signed_cfg.get("negative_source", "rolled")),
		"generated_omega": float(signed_cfg.get("generated_omega", 0.8)),
		"schedule": resolve_signed_schedule(signed_cfg.get("schedule", None), float(signed_cfg.get("alpha", 0.0))),
	}
	amortised_cfg = signed_cfg.get("amortised", None) or {}
	if isinstance(amortised_cfg, bool):
		amortised_cfg = {"enabled": bool(amortised_cfg)}
	resolved["amortised"] = {
		"enabled": bool(amortised_cfg.get("enabled", False)),
		"alpha_max": float(amortised_cfg.get("alpha_max", 3.0)),
		"beta": float(amortised_cfg.get("beta", 1.0)),
		"zero_prob": float(amortised_cfg.get("zero_prob", 0.1)),
	}
	if resolved["alpha"] < 0.0 or not math.isfinite(resolved["alpha"]):
		raise ValueError("signed.alpha must be finite and >= 0")
	if resolved["enabled"] and resolved["alpha"] == 0.0 and resolved["schedule"] is None and not resolved["amortised"]["enabled"]:
		raise ValueError(
			"signed.enabled is set but signed.alpha is 0, which is the native rule. "
			"Set signed.alpha > 0, or signed.enabled=false."
		)
	if resolved["paired"] and not resolved["enabled"]:
		raise ValueError("signed.paired is the signed null control; it needs signed.enabled")
	return resolved



REMOVED_LOSSES = ("beckmann_transport", "imf", "autonomous_drift", "map_distillation", "fixed_t_regression")


def get_loss_fn(cfg):
	loss_name = cfg.loss.get("name", "guided_cbm")
	if loss_name == "guided_cbm":
		# the self-guided objective (guided transport + endpoint from a frozen teacher), sbm_rae/guided_cbm.py
		from sbm_rae.guided_cbm import get_guided_cbm_loss_fn

		return get_guided_cbm_loss_fn(cfg)
	if loss_name == "bridge_cbm":
		# the same objective on the Bridge-V2 / RT-1 world model's condition dict (sbm_rae/bridge_loss.py)
		from sbm_rae.bridge_loss import get_bridge_cbm_loss_fn

		return get_bridge_cbm_loss_fn(cfg)
	if loss_name in REMOVED_LOSSES:
		raise ValueError(
			f"loss.name={loss_name!r} was removed from this code; "
			"available: guided_cbm, bridge_cbm"
		)
	raise ValueError(f"Unsupported loss: {loss_name}")
