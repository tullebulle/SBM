import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.rae.eval_fid as eval_fid
from sbm_rae.ema import ema_decay_label


def _fid_cfg(*, skip_existing=True, ema_decays=None, strict_ema_decays=False):
    return OmegaConf.create(
        {
            "fid": {
                "skip_existing": skip_existing,
                "ema_decays": [] if ema_decays is None else list(ema_decays),
                "strict_ema_decays": strict_ema_decays,
            }
        }
    )


def _distributed():
    return eval_fid.DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1)


def test_load_or_evaluate_checkpoint_skips_valid_existing_metrics(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    expected = {"fid": 12.5, "checkpoint_path": "model.ckpt"}
    (output_dir / "fid_metrics.json").write_text(json.dumps(expected) + "\n")

    def fail_evaluate(*args, **kwargs):
        raise AssertionError("existing FID metrics should skip evaluation")

    monkeypatch.setattr(eval_fid, "_evaluate_checkpoint", fail_evaluate)

    result = eval_fid._load_or_evaluate_checkpoint(
        _fid_cfg(skip_existing=True),
        Path("model.ckpt"),
        output_dir,
        torch.device("cpu"),
        _distributed(),
        eval_fid.WeightVariant(source="active"),
    )

    assert result == expected


def test_load_or_evaluate_checkpoint_recomputes_invalid_existing_metrics(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "fid_metrics.json").write_text("{not valid json")
    expected = {"fid": 3.25, "checkpoint_path": "model.ckpt"}
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(eval_fid, "_evaluate_checkpoint", fake_evaluate)

    result = eval_fid._load_or_evaluate_checkpoint(
        _fid_cfg(skip_existing=True),
        Path("model.ckpt"),
        output_dir,
        torch.device("cpu"),
        _distributed(),
        eval_fid.WeightVariant(source="active"),
    )

    assert result == expected
    assert len(calls) == 1


def test_weight_variants_include_active_and_requested_inactive_ema(monkeypatch):
    active_decay = 0.9999
    inactive_decay = 0.999
    monkeypatch.setattr(
        eval_fid,
        "_load_checkpoint",
        lambda checkpoint_path: {
            "ema_active_decay": active_decay,
            "ema_state_dicts": {ema_decay_label(inactive_decay): {}},
        },
    )

    variants = eval_fid._weight_variants_for_checkpoint(
        _fid_cfg(
            ema_decays=[active_decay, inactive_decay],
            strict_ema_decays=True,
        ),
        Path("model.ckpt"),
        _distributed(),
    )

    assert variants == [
        eval_fid.WeightVariant(
            source="active",
            label=ema_decay_label(active_decay),
            decay=active_decay,
        ),
        eval_fid.WeightVariant(
            source="ema_state_dicts",
            label=ema_decay_label(inactive_decay),
            decay=inactive_decay,
        ),
    ]


def test_weight_variants_fall_back_to_saved_weights_when_nothing_is_requested(monkeypatch):
    monkeypatch.setattr(
        eval_fid,
        "_load_checkpoint",
        lambda checkpoint_path: {"ema_active_decay": 0.999, "ema_state_dicts": {}},
    )

    variants = eval_fid._weight_variants_for_checkpoint(
        _fid_cfg(ema_decays=[]),
        Path("model.ckpt"),
        _distributed(),
    )

    # An empty list here used to mean zero evaluations and an empty summary.
    assert variants == [
        eval_fid.WeightVariant(source="active", label=ema_decay_label(0.999), decay=0.999)
    ]


def test_build_eval_cfg_merges_model_overrides_from_the_fid_config(monkeypatch, tmp_path):
    """`+model.attn_func=default` must reach the model the scorer builds.

    Without this the model is rebuilt purely from the checkpoint's training
    config, so a Hopper-trained checkpoint cannot be scored on an A100 at all.
    Only the overridden key changes; the rest of the model config is the
    checkpoint's.
    """
    import scripts.rae.eval_fid as eval_fid
    from omegaconf import OmegaConf

    training_cfg = OmegaConf.create({
        "model": {"name": "dit_b_2", "attn_func": "fa3", "depth": 12},
        "dataset": {"name": "cifar10", "img_resolution": 32},
        "data_dir": str(tmp_path),
        "seed": 7,
    })
    monkeypatch.setattr(eval_fid, "_load_training_cfg", lambda path: training_cfg)
    fid_cfg = OmegaConf.create({
        "fid": {"real_split": "train"},
        "real_dir": str(tmp_path),
        "model": {"attn_func": "default"},
    })

    eval_cfg = eval_fid._build_eval_cfg(fid_cfg, tmp_path / "x.ckpt", tmp_path)

    assert eval_cfg.model.attn_func == "default"
    assert eval_cfg.model.name == "dit_b_2"
    assert eval_cfg.model.depth == 12

    plain = eval_fid._build_eval_cfg(
        OmegaConf.create({"fid": {"real_split": "train"}, "real_dir": str(tmp_path)}),
        tmp_path / "x.ckpt", tmp_path,
    )
    assert plain.model.attn_func == "fa3"


def test_raw_weight_variant_reads_current_model_state():
    """A checkpoint's state_dict is the ACTIVE EMA copy; raw weights live elsewhere.

    Scoring raw is how a gap between two models is separated into learning versus
    averaging, so the variant must read current_model_state and must say so loudly
    when a checkpoint does not carry it.
    """
    import importlib.util, sys
    import torch

    spec = importlib.util.spec_from_file_location(
        "_ef", Path(__file__).resolve().parents[2] / "scripts/rae/eval_fid.py"
    )
    mod = importlib.util.module_from_spec(spec); sys.modules["_ef"] = mod
    spec.loader.exec_module(mod)

    variant = mod.WeightVariant(source="raw", label="raw")
    assert variant.suffix == "_raw"
    assert mod.WeightVariant(source="active").suffix == ""

    path = Path(__file__).resolve().parents[2] / "outputs" / "_test_raw_variant.ckpt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": {"a": torch.ones(2)},
                "current_model_state": {"a": torch.zeros(2)}}, path)
    try:
        raw = mod._load_raw_state_dict(path)
        torch.testing.assert_close(raw["a"], torch.zeros(2))
        torch.save({"state_dict": {"a": torch.ones(2)}}, path)
        with pytest.raises(ValueError, match="current_model_state"):
            mod._load_raw_state_dict(path)
    finally:
        path.unlink(missing_ok=True)


def test_score_raw_flag_adds_the_variant():
    import importlib.util, sys
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "_ef2", Path(__file__).resolve().parents[2] / "scripts/rae/eval_fid.py"
    )
    mod = importlib.util.module_from_spec(spec); sys.modules["_ef2"] = mod
    spec.loader.exec_module(mod)

    import torch

    path = Path(__file__).resolve().parents[2] / "outputs" / "_test_score_raw_flag.ckpt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": {"a": torch.ones(2)},
                "current_model_state": {"a": torch.zeros(2)},
                "ema_decays": {"0_999": 0.999}, "ema_active_decay": 0.999,
                "ema_state_dicts": {"0_999": {"a": torch.ones(2)}}}, path)
    try:
        cfg = OmegaConf.create({"fid": {"ema_decays": [], "score_raw": True}})
        variants = mod._weight_variants_for_checkpoint(cfg, str(path), False)
        assert any(v.source == "raw" for v in variants), variants
        cfg_off = OmegaConf.create({"fid": {"ema_decays": [], "score_raw": False}})
        assert not any(
            v.source == "raw"
            for v in mod._weight_variants_for_checkpoint(cfg_off, str(path), False)
        )
    finally:
        path.unlink(missing_ok=True)
