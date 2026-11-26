from collections.abc import Iterable
from numbers import Real

import pandas as pd


def as_float(value: object, *, label: str) -> float:
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"{label} is not a real number")


def expect[T](value: object, cls: type[T]) -> T:
    if not isinstance(value, cls):
        raise TypeError(f"expected {cls.__name__}, got {type(value).__name__}")
    return value


def expect_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return frame
