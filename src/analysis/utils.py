from collections.abc import Iterable

import pandas as pd


def expect[T](value: object, cls: type[T]) -> T:
    if not isinstance(value, cls):
        raise TypeError(f"expected {cls.__name__}, got {type(value).__name__}")
    return value


def expect_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return frame
