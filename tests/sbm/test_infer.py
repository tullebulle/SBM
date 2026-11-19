"""Exports (sbm.infer): `sbm eval <export>` reads with the export's own settings unless the command line overrides them."""
from __future__ import annotations

import torch

from sbm import eval as E
from sbm.infer import FORMAT
from sbm.models import build_model


def _export(path, read):
    m = build_model("DiT-tiny", input_size=8, label_dim=10)
    torch.save({"format": FORMAT, "state_dict": {f"model.{k}": v for k, v in m.state_dict().items()},
                "model": {"name": "DiT-tiny", "input_size": 8, "label_dim": 10}, "cfg": {}, "ema_decay": 0.9999,
                "global_step": 1, "latents_scale": 0.18215, "meta": {"model": "tiny", "read": read, "results": {}}}, path)
    return path


def _calls(monkeypatch, tmp_path):
    seen = []

    def dispatch(protocol, model, p, device, scale):
        seen.append((protocol, dict(p)))
        return {"fid_adm": {"1": 1.0}}, {}, 0.0
    monkeypatch.setenv("SBM_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(E, "_dispatch", dispatch)
    return seen


def test_export_eval_defaults_to_its_read(tmp_path, monkeypatch):
    seen = _calls(monkeypatch, tmp_path)
    f = _export(tmp_path / "x.pt", {"protocol": "xl_val50k", "omega": 0.95, "label_scale": 1.2, "ema": 0.9999, "step": 1})
    E.evaluate(f, [])
    protocol, p = seen[-1]
    assert protocol == "xl_val50k" and (p["omega"], p["label_scale"], p["weights"]) == (0.95, 1.2, "ema") and p["bf16_weights"]
    E.evaluate(f, ["omega=0.9", "protocol=b2_adm50k"])
    protocol, p = seen[-1]
    assert protocol == "b2_adm50k" and (p["omega"], p["label_scale"]) == (0.9, 1.2) and "bf16_weights" not in p


def test_checkpoint_eval_keeps_the_protocol_defaults(tmp_path, monkeypatch):
    seen = _calls(monkeypatch, tmp_path)
    m = build_model("DiT-tiny", input_size=8, label_dim=10)
    f = tmp_path / "plain.pt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in m.state_dict().items()}, "model_name": "DiT-tiny",
                "hyper_parameters": {"cfg": {"model": {"name": "DiT-tiny", "input_size": 8, "label_dim": 10}}}}, f)
    E.evaluate(f, [])
    protocol, p = seen[-1]
    assert protocol == "b2_adm50k" and p["omega"] == 0.8 and "label_scale" not in p
