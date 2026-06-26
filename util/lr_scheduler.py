# -*- coding: utf-8 -*-
""" Scheduler for optimizer. 
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def CosineAnnealingWithWarmup(
    optimizer: Optimizer,
    warmup_steps: int = 0,
    cycle_steps: int = 1000,
    lr_min: float = 0.0,
    lr_max: float = 0.1,
) -> LambdaLR:
    """
    Step-based scheduler with linear warm-up followed by **periodic** cosine annealing.
    After warm-up, the cosine curve repeats every `cycle_steps` iterations.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warm-up iterations (linear 0 -> lr_max).
        cycle_steps (int): Length of one cosine annealing period (must > 0).
        lr_min (float): Minimum learning rate after annealing.
        lr_max (float): Maximum (initial) learning rate.

    Returns:
        LambdaLR: Call `.step()` every iteration.
    """
    assert cycle_steps > 0, "cycle_steps must be positive"

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Periodic cosine annealing: progress wraps every cycle_steps
            progress = (step - warmup_steps) % cycle_steps / cycle_steps
            coeff = 0.5 * (1 + math.cos(math.pi * progress))
            return lr_min / lr_max + (1 - lr_min / lr_max) * coeff

    return LambdaLR(optimizer, lr_lambda)