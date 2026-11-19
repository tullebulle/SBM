"""Checkpointing that does not block training on a shared filesystem.

A multi-GB checkpoint write to a contended shared filesystem can stall for tens
of minutes *inside* the collective barrier that follows the save. The other ranks sit in an allreduce,
the NCCL watchdog eventually fires, and what is really a filesystem stall is
reported as a distributed failure.

The fix is to write to node-local scratch, which is a local block device and
takes a couple of seconds, and copy to the shared run directory on a background
thread afterwards. Training continues during the copy, so a slow shared
filesystem costs throughput at worst instead of killing the run.

Only rank 0 writes checkpoints, so only rank 0 does any of this.
"""

import os
import shutil
import threading
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint


class LocalThenSyncCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that saves locally and mirrors to shared storage."""

    def __init__(self, *args, shared_dirpath=None, local_dirpath=None, sparse_after=None, sparse_every=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_dirpath = Path(shared_dirpath) if shared_dirpath else None
        self.local_dirpath = Path(local_dirpath) if local_dirpath else None
        self.sparse_after = int(sparse_after or 0)
        self.sparse_every = int(sparse_every or 0)
        self._threads = []

    def _should_skip_saving_checkpoint(self, trainer):
        # Dense early, sparse late: past `sparse_after` only every `sparse_every`-th step is saved. Every rank sees the
        # same global step, so they all skip together.
        step = trainer.global_step
        if self.sparse_every and step > self.sparse_after and step % self.sparse_every != 0:
            return True
        return super()._should_skip_saving_checkpoint(trainer)

    def _mirror(self, local_path, shared_path):
        try:
            tmp = shared_path.with_suffix(shared_path.suffix + ".partial")
            shutil.copyfile(local_path, tmp)
            os.replace(tmp, shared_path)          # atomic: readers see whole files
        except Exception as exc:                  # never let mirroring kill a run
            print(f"checkpoint mirror to {shared_path} failed: {exc}")
            return
        self._prune_local(keep=local_path)

    def _prune_local(self, keep):
        # Node-local scratch is only a staging area: kept forever, the checkpoints of a long run fill /tmp and kill the
        # run at its next save. Once a checkpoint is mirrored, every other local copy whose shared copy is complete
        # (exists, same size) goes; one not yet mirrored stays.
        for path in self.local_dirpath.glob("*.ckpt"):
            if path == keep or path.is_symlink():
                continue
            shared = self.shared_dirpath / path.name
            try:
                if shared.is_file() and shared.stat().st_size == path.stat().st_size:
                    path.unlink()
            except OSError as exc:
                print(f"local checkpoint cleanup of {path} failed: {exc}")

    def _save_checkpoint(self, trainer, filepath):
        if self.local_dirpath is None or self.shared_dirpath is None:
            return super()._save_checkpoint(trainer, filepath)

        name = Path(filepath).name
        local_path = self.local_dirpath / name
        self.local_dirpath.mkdir(parents=True, exist_ok=True)

        # The expensive, barrier-adjacent part goes to local disk.
        super()._save_checkpoint(trainer, str(local_path))

        if trainer.is_global_zero:
            shared_path = self.shared_dirpath / name
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            thread = threading.Thread(
                target=self._mirror, args=(local_path, shared_path), daemon=True
            )
            thread.start()
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]

    def teardown(self, trainer, pl_module, stage):
        # Let in-flight mirrors finish so the last checkpoint reaches shared
        # storage before the job exits.
        for thread in self._threads:
            thread.join(timeout=600)
        return super().teardown(trainer, pl_module, stage)
