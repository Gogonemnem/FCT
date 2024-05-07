from torch.optim import Optimizer


class GradientAccumulatorWrapper:
    def __init__(self, optimizer: Optimizer, accumulation_steps=1):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def zero_grad(self):
        self.step_count += 1
        if self.step_count >= self.accumulation_steps:
            self.step_count = 0
            self.optimizer.zero_grad()

    def step(self):
        if self.step_count < self.accumulation_steps:
            self.optimizer.step()