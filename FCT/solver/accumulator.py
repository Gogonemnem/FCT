import torch
from torch.optim import Optimizer

class GradientAccumulator(Optimizer):
    def __init__(self, optimizer: Optimizer, accumulation_steps=1):
        # Initialize the base optimizer
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def zero_grad(self, set_to_none: bool = False):
        if self.step_count == 0:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.step_count += 1
        if self.step_count >= self.accumulation_steps:
            self.optimizer.step(closure=closure)
            self.zero_grad()
            self.step_count = 0

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults