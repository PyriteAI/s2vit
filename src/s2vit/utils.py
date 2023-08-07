from collections.abc import Sequence
from typing import Any


def to_pair(x: Any) -> tuple[Any, Any]:
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        if len(x) == 0 or len(x) > 2:
            raise ValueError(f"expected a sequence of length 1 or 2, got {len(x)}")
        if len(x) == 1:
            return x[0], x[0]
        return tuple(x)
    return x, x


__all__ = ["to_pair"]
