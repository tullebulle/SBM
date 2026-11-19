"""The inference export (sbm_t2i.infer): one EMA copy + the run config, read by load_model, by load_weights (init.from)
and by the GenEval sampler like a training checkpoint."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from sbm_t2i.infer import FORMAT, READS, ema_copy, export_from_checkpoint, is_export, load_model, read_settings
from sbm_t2i.models import build_model, load_weights

KW = dict(name="TextDiT-tiny", input_size=8, in_channels=6, txt_dim=16, txt_len=8, text_depth=1)


def fake_checkpoint(tmp_path):
    """A training checkpoint's layout: active EMA (0.999) in state_dict, 0.9999 in ema_state_dicts, raw weights, loss state."""
    torch.manual_seed(0)
    model = build_model(**KW)
    sd = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(1)
    other = {k: v + 0.01 * torch.randn(v.shape, generator=g) for k, v in sd.items()}
    ck = {"state_dict": sd, "ema_state_dicts": {"0_9999": other}, "ema_active_decay": 0.999,
          "current_model_state": {k: v - 1.0 for k, v in sd.items()}, "global_step": 1234,
          "guided_cbm_loss": {"ratio_ema": 3.0, "frozen_weight": 2.5}, "optimizer_states": [{"not": "exported"}]}
    path = tmp_path / "step0001234.ckpt"
    torch.save(ck, path)
    return path, ck


def test_export_holds_one_ema_copy_and_the_loss_state(tmp_path):
    path, ck = fake_checkpoint(tmp_path)
    cfg = {"model": KW, "codec": {"kind": "rae", "stats": "${store:}/rae/assets/rae/stats.pt"}}
    exp = export_from_checkpoint(path, cfg, which="0.9999", meta={"note": "test"})
    assert exp["format"] == FORMAT and is_export(exp) and not is_export(ck)
    assert exp["ema_active_decay"] == 0.9999 and exp["global_step"] == 1234 and exp["meta"] == {"note": "test"}
    assert exp["guided_cbm_loss"] == ck["guided_cbm_loss"] and "optimizer_states" not in exp
    assert exp["cfg"]["codec"]["stats"] == "${store:}/rae/assets/rae/stats.pt"   # interpolations kept unresolved
    for k, v in ck["ema_state_dicts"]["0_9999"].items():
        assert torch.equal(exp["state_dict"][k], v)
    assert ema_copy(ck, "0.999") is ck["state_dict"] and ema_copy(ck, "raw") is ck["current_model_state"]


def test_export_loads_like_a_checkpoint(tmp_path):
    path, ck = fake_checkpoint(tmp_path)
    out = tmp_path / "export.pt"
    torch.save(export_from_checkpoint(path, {"model": KW}, which="0.9999"), out)
    assert is_export(out) and not is_export(path)
    model, cfg, step = load_model(out, "0.9999")
    assert step == 1234 and cfg.model.name == "TextDiT-tiny"
    ref = {k[len("model."):]: v for k, v in ck["ema_state_dicts"]["0_9999"].items()}
    assert all(torch.equal(v, ref[k]) for k, v in model.state_dict().items())
    via_init = load_weights(build_model(**KW), out, "ema")          # init.from=<export> (init.resume_state=false)
    assert all(torch.equal(v, ref[k]) for k, v in via_init.state_dict().items())
    with pytest.raises(ValueError, match="EMA 0.9999 only"):
        load_model(out, "0.999")


def test_checkpoint_path_reads_the_run_config(tmp_path):
    path, ck = fake_checkpoint(tmp_path)
    run = tmp_path / "run"
    (run / "ckpt").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"model": KW}), run / "run.yaml")
    target = run / "ckpt" / path.name
    path.rename(target)
    model, cfg, step = load_model(target, "0.999")                  # the active EMA: state_dict
    assert step == 1234
    assert all(torch.equal(v, ck["state_dict"]["model." + k]) for k, v in model.state_dict().items())


def test_read_settings_default_to_the_export_meta(tmp_path):
    path, ck = fake_checkpoint(tmp_path)
    out = tmp_path / "export.pt"
    torch.save(export_from_checkpoint(path, {"model": KW}, which="0.9999", meta={"read": {"omega": 0.4, "label_scale": 1.1}}), out)
    _, cfg, _ = load_model(out, "0.9999")
    assert cfg.read == {"omega": 0.4, "label_scale": 1.1}
    torch.save(export_from_checkpoint(path, {"model": KW}, which="0.9999"), out)
    assert load_model(out, "0.9999")[1].read == {"omega": 0.8, "label_scale": 1.0}       # no meta, unknown network
    xl = OmegaConf.create({"model": {"name": "TextDiT-XL/1-DH"}})
    assert read_settings(xl) == READS["TextDiT-XL/1-DH"] == {"omega": 0.35, "label_scale": 1.125}
    assert read_settings(OmegaConf.create({"model": {"name": "TextDiT-B/1-DH-t15"}})) == {"omega": 0.3, "label_scale": 1.25}
