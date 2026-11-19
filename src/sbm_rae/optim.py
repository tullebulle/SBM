import torch
from torch.optim import Optimizer


class ChainedOptimizer(Optimizer):
    def __init__(self, optimizers):
        self.optimizers = tuple(optimizers)
        if not self.optimizers:
            raise ValueError("ChainedOptimizer requires at least one optimizer")

        param_groups = [
            param_group
            for optimizer in self.optimizers
            for param_group in optimizer.param_groups
        ]
        super().__init__(param_groups, defaults={})
        self.param_groups = param_groups
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss

    def state_dict(self):
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers]
        }

    def load_state_dict(self, state_dict):
        optimizer_states = state_dict.get("optimizers", None)
        if optimizer_states is None:
            raise ValueError("Missing optimizer states for ChainedOptimizer")
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "Loaded optimizer state count does not match ChainedOptimizer "
                f"({len(optimizer_states)} != {len(self.optimizers)})"
            )

        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)

        self.param_groups = [
            param_group
            for optimizer in self.optimizers
            for param_group in optimizer.param_groups
        ]
        self.state = {}