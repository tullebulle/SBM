"""Run folders: every training run lives in exactly one folder in the store.

    <store>/runs/<YYYYMMDD-HHMMSS>_<name>/
        run.yaml     resolved config + provenance: commit, branch, dirty flag, command, host, versions
        git.diff     uncommitted changes at launch (only when the tree was dirty)
        ckpt/        checkpoints (step*.ckpt; a run resumes from the newest)
        logs/        free for the caller's logs
        evals/       one JSON per evaluation, each carrying its protocol (samples, reference, EMA, seed, NFE, omega)
        wandb/       the wandb run (wandb.mode=offline or disabled to keep it local)

To reproduce a run: `python -m sbm_t2i.cli reproduce <run_dir>` checks out its commit, applies git.diff and trains with
run.yaml.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STORE = str(REPO / "store")   # data, weights and runs live here unless SBM_STORE points elsewhere


def store() -> Path:
    return Path(os.environ.get("SBM_STORE") or (REPO / "store"))


# `${store:}` / `${repo:}` in a config resolve to the store / the repository at resolution time (the same `store`
# resolver as the sbm package's); registered on `import sbm_t2i`
OmegaConf.register_new_resolver("store", lambda: str(store()), replace=True)
OmegaConf.register_new_resolver("repo", lambda: str(REPO), replace=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


def git_state() -> dict:
    diff = _git("diff", "HEAD")
    return dict(commit=_git("rev-parse", "HEAD"), branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
                remote=_git("remote", "get-url", "origin"), dirty=bool(diff), diff=diff)


def provenance() -> dict:
    import torch
    env = dict(host=socket.gethostname(), python=platform.python_version(), torch=str(torch.__version__),
               cuda=str(torch.version.cuda), argv=sys.argv, cwd=os.getcwd())
    if torch.cuda.is_available():
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return env


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    if not s:
        raise ValueError("a run needs a name (name=...)")
    return s


def new_run(cfg: DictConfig) -> Path:
    """Create the run folder, freeze the resolved config and the code state into run.yaml, return the folder."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = store() / "runs" / f"{stamp}_{slug(cfg.name)}"
    for sub in ("ckpt", "logs", "evals", "wandb"):
        (run_dir / sub).mkdir(parents=True, exist_ok=False if sub == "ckpt" else True)
    git = git_state()
    if git["dirty"]:
        (run_dir / "git.diff").write_text(git["diff"])
        print(f"warning: the tree is dirty; the diff is saved to {run_dir / 'git.diff'}", file=sys.stderr)
    frozen = OmegaConf.to_container(cfg, resolve=True)
    frozen["run"] = dict(dir=str(run_dir), created=stamp,
                         git={k: v for k, v in git.items() if k != "diff"}, env=provenance())
    OmegaConf.save(OmegaConf.create(frozen), run_dir / "run.yaml")
    return run_dir


def load_run(run_dir: str | Path) -> DictConfig:
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "run.yaml")
    head = git_state()
    if head["commit"] != cfg.run.git.commit:
        print(f"warning: run.yaml was frozen at {cfg.run.git.commit[:10]}, the tree is at {head['commit'][:10]}; "
              f"`python -m sbm_t2i.cli reproduce {run_dir}` runs the exact code", file=sys.stderr)
    return cfg


def wandb_kwargs(cfg: DictConfig, run_dir: Path) -> dict:
    """wandb for every run: its name and id are the folder name, so a resumed run appends to the same wandb run."""
    return dict(project=cfg.wandb.project, entity=cfg.wandb.get("entity"), name=run_dir.name,
                id=re.sub(r"[^A-Za-z0-9_-]", "-", run_dir.name), save_dir=str(run_dir / "wandb"), resume="allow",
                mode=cfg.wandb.get("mode", "online"), tags=list(cfg.wandb.get("tags", [])),
                notes=f"commit {cfg.run.git.commit[:10]}", config=OmegaConf.to_container(cfg, resolve=True))


def init_wandb(cfg: DictConfig, run_dir: Path):
    import wandb
    kw = wandb_kwargs(cfg, run_dir)
    kw["dir"] = kw.pop("save_dir")
    run = wandb.init(**kw)
    (run_dir / "wandb" / "url.txt").write_text(str(getattr(run, "url", "") or ""))
    return run


def record_eval(run_dir: str | Path, name: str, result: dict) -> Path:
    """One JSON per evaluation; `result` must carry its protocol (see sbm_t2i.eval)."""
    p = Path(run_dir) / "evals" / f"{name}.json"
    p.write_text(json.dumps(result, indent=1))
    return p
