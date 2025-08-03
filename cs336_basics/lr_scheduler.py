import math


def learning_rate_schedule(
    t: int,
    amax: float,
    amin: float,
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
        return amax * t / warmup_steps
    elif (t >= warmup_steps) and (t <= decay_steps):
        decay = 0.5 * (1 + math.cos(math.pi * (t - warmup_steps) / (decay_steps - warmup_steps)))
        return amin + (amax - amin) * decay
    else:
        return amin