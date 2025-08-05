import math


def learning_rate_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """
    Compute the learning rate at step `t` using a cosine decay schedule with warmup.

    Args:
        t (int): Current step.
        amax (float): Maximum learning rate.
        amin (float): Minimum learning rate.
        warmup_steps (int): Number of steps for warmup.
        decay_steps (int): Total number of steps for decay.

    Returns:
        float: Learning rate at step `t`.
    """
    if t < warmup_steps:
        return max_lr * t / warmup_steps
    elif (t >= warmup_steps) and (t <= decay_steps):
        decay = 0.5 * (1 + math.cos(math.pi * (t - warmup_steps) / (decay_steps - warmup_steps)))
        return min_lr + (max_lr - min_lr) * decay
    else:
        return min_lr