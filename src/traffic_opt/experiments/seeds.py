"""Seed utilities for reproducible traffic experiments."""

DEFAULT_SCENARIO_SEED = 20260919


def validate_seed(seed: int) -> int:
    """Return a valid non-negative integer seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("scenario seed must be a non-negative integer")
    return seed


def derive_seed(seed: int, stream: int) -> int:
    """Derive a deterministic independent stream seed."""

    validate_seed(seed)
    if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
        raise ValueError("seed stream must be a non-negative integer")
    return (seed * 1_000_003 + stream * 97_003) % (2**32)