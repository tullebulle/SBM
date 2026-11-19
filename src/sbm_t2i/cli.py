"""The command line.

    python -m sbm_t2i.cli train EXP [overrides...]     create a run folder (config + code state frozen) and train it
    python -m sbm_t2i.cli train RUN_DIR                train (or resume) an existing run folder from its newest checkpoint
    python -m sbm_t2i.cli new   EXP [overrides...]     only create the run folder and print its path
    python -m sbm_t2i.cli eval  RUN_DIR|CKPT [opts]    class-conditional FID with a named protocol into <run>/evals/
    python -m sbm_t2i.cli reproduce RUN_DIR            check out the run's commit (+ git.diff) in a worktree and train its
                                                       frozen config again as a new run

EXP names a file in configs/t2i/exp/ (e.g. t2i_rae_xl1dh) or a frozen run.yaml; overrides are Hydra-style key=value
pairs. Without name=..., the run is named after the experiment.

Multi-GPU: on one node, `python -m sbm_t2i.cli train ...` trains on every visible GPU (Lightning starts one process
per GPU; train.devices=N for fewer). With torchrun, create the run folder first and hand every process the folder:

    RUN=$(python -m sbm_t2i.cli new EXP [overrides...] | tail -1)
    torchrun --nproc_per_node N [--nnodes M --node_rank R --master_addr HOST] -m sbm_t2i.cli train $RUN
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from sbm_t2i import run as R

CONFIGS = R.REPO / "configs" / "t2i"


def compose_exp(exp: str, overrides: list[str]) -> DictConfig:
    """configs/t2i/train.yaml + configs/t2i/exp/<exp>.yaml + overrides, or a frozen run.yaml + overrides."""
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


# the run folder a parent process created ("<parent pid>:<folder>"): the per-GPU processes Lightning starts re-run this
# command line and must train in the parent's folder instead of creating their own
RUN_ENV = "SBM_T2I_RUN_DIR"


def _parent_run_dir():
    pid, _, path = os.environ.get(RUN_ENV, "").partition(":")
    return Path(path) if pid.isdigit() and int(pid) == os.getppid() and path else None


def cmd_new(a):
    print(R.new_run(compose_exp(a.target, a.overrides)))


def cmd_train(a):
    from sbm_t2i.train import train
    if Path(a.target).is_dir():
        run_dir = Path(a.target)
        cfg = R.load_run(run_dir)
    elif _parent_run_dir() is not None:
        run_dir = _parent_run_dir()
        cfg = OmegaConf.load(run_dir / "run.yaml")
    else:
        if "TORCHELASTIC_RUN_ID" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise SystemExit("under torchrun, create the run folder first (python -m sbm_t2i.cli new EXP ...) and pass it: "
                             "train RUN_DIR")
        run_dir = R.new_run(compose_exp(a.target, a.overrides))
        os.environ[RUN_ENV] = f"{os.getpid()}:{run_dir}"
        cfg = OmegaConf.load(run_dir / "run.yaml")
    train(cfg, run_dir)


def cmd_eval(a):
    from sbm_t2i.eval import evaluate
    evaluate(a.target, a.overrides)


def cmd_reproduce(a):
    run_dir = Path(a.target).resolve()
    cfg = OmegaConf.load(run_dir / "run.yaml")
    wt = R.store() / "worktrees" / cfg.run.git.commit[:12]
    if not wt.exists():
        subprocess.run(["git", "-C", str(R.REPO), "worktree", "add", "--detach", str(wt), cfg.run.git.commit], check=True)
        if (run_dir / "git.diff").exists():
            subprocess.run(["git", "-C", str(wt), "apply", str(run_dir / "git.diff")], check=True)
    print(f"code at {wt}; the frozen config of {run_dir.name} runs again as a new run")
    env = dict(os.environ, PYTHONPATH=str(wt / "src"))
    cmd = [sys.executable, "-m", "sbm_t2i.cli", "train", str(run_dir / "run.yaml"), f"name={cfg.name}-repro"]
    os.chdir(wt)
    os.execvpe(sys.executable, cmd, env)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sbm_t2i.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("new", cmd_new), ("eval", cmd_eval)):
        p = sub.add_parser(name)
        p.add_argument("target")
        p.add_argument("overrides", nargs="*")
        p.set_defaults(fn=fn)
    p = sub.add_parser("reproduce"); p.add_argument("target")
    p.set_defaults(fn=cmd_reproduce)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
