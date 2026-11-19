"""The command line of the SD-VAE-latent trainer.

    python -m sbm.cli train  EXP [overrides...]    create a run folder and train in it, on every GPU of this node
    python -m sbm.cli train  RUN_DIR               train (or resume) an existing run folder
    python -m sbm.cli new    EXP [overrides...]    create the run folder only and print its path (for torchrun):
        RUN=$(python -m sbm.cli new b2_sbm name=b2) && torchrun --nproc_per_node 4 -m sbm.cli train $RUN
    python -m sbm.cli eval   RUN_DIR|CKPT|EXPORT [opts]   FID-50k with a named protocol (sbm.eval)
    python -m sbm.cli config EXP [overrides...]    print the composed, resolved config (no run folder is made)

EXP names a file in configs/sbm/exp/ (e.g. b2_sbm, xl_sbm, xl_finetune) or a frozen run.yaml; overrides are key=value
pairs (Hydra syntax). Without name=..., the run is named after the experiment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from sbm import run as R

CONFIGS = R.REPO / "configs" / "sbm"


def compose_exp(exp: str, overrides: list[str]) -> DictConfig:
    """configs/sbm/train.yaml + configs/sbm/exp/<exp>.yaml + overrides, or a frozen run.yaml + overrides."""
    if exp.endswith((".yaml", ".yml")):
        cfg = OmegaConf.load(exp)
        cfg.pop("run", None)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    else:
        with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
            cfg = compose(config_name="train", overrides=[f"+exp={exp}", *overrides])
    if OmegaConf.is_missing(cfg, "name"):
        cfg.name = cfg.get("exp_name") or Path(exp).stem
    # an experiment marks the inputs it cannot default as '<required>' (a '???' would not survive the merge onto train.yaml)
    flat = OmegaConf.to_container(cfg, resolve=False)
    missing = sorted(OmegaConf.missing_keys(cfg)) + sorted(_required(flat))
    if missing:
        raise SystemExit(f"{exp} needs: {', '.join(k + '=...' for k in missing)}")
    return cfg


def _required(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _required(v, f"{prefix}{k}.")
        elif v == "<required>":
            yield f"{prefix}{k}"


def cmd_new(a):
    print(R.new_run(compose_exp(a.target, a.overrides)))


def cmd_train(a):
    """A run folder trains as is. An experiment gets a new run folder first; the per-GPU processes Lightning starts for
    a multi-GPU run re-execute this command and find that folder through SBM_RUN_DIR instead of making their own."""
    from sbm.train import train
    if Path(a.target).is_dir():
        run_dir = Path(a.target)
    elif os.environ.get("SBM_RUN_DIR"):
        run_dir = Path(os.environ["SBM_RUN_DIR"])
    else:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise SystemExit("under torchrun, create the run folder first: RUN=$(python -m sbm.cli new EXP ...), then train $RUN")
        run_dir = R.new_run(compose_exp(a.target, a.overrides))
        print(run_dir, flush=True)
        os.environ["SBM_RUN_DIR"] = str(run_dir)   # inherited by the per-GPU processes Lightning starts
        try:
            return train(R.load_run(run_dir), run_dir)
        finally:
            os.environ.pop("SBM_RUN_DIR", None)
    train(R.load_run(run_dir), run_dir)


def cmd_eval(a):
    from sbm.eval import evaluate
    evaluate(a.target, a.overrides)


def cmd_config(a):
    """The composed config, resolved (`${store:}` and friends filled in), as the run.yaml would freeze it."""
    cfg = compose_exp(a.target, a.overrides)
    print(OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))), end="")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sbm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("new", cmd_new), ("eval", cmd_eval), ("config", cmd_config)):
        p = sub.add_parser(name)
        p.add_argument("target")
        p.add_argument("overrides", nargs="*")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
