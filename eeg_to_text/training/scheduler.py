"""
Learning Rate and Teacher-Forcing Schedulers
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


# ════════════════════════════════════════════════════════════════════════════
# Cosine LR schedule with linear warmup
# ════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Cosine annealing from peak LR down to min_lr_ratio × peak_LR,
    with a linear warmup over the first ``num_warmup_steps`` steps.

    Args:
        optimizer:           The optimiser.
        num_warmup_steps:    Steps for linear warmup.
        num_training_steps:  Total training steps (warmup + decay).
        min_lr_ratio:        Minimum LR as a fraction of peak LR.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════

