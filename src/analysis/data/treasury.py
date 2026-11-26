from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis import columns as col, utils


DAYS_PER_YEAR: float = 365.25


@dataclass(frozen=True)
class ZeroTenor:
    name: str
    years: float
    target_yield: str
    lower_yield: str


ZERO_TENORS: tuple[ZeroTenor, ...] = (
    ZeroTenor(col.UST_ZERO_3Y, 3.0, col.GSW_ZERO_3Y, col.GSW_ZERO_2Y),
    ZeroTenor(col.UST_ZERO_5Y, 5.0, col.GSW_ZERO_5Y, col.GSW_ZERO_4Y),
    ZeroTenor(col.UST_ZERO_10Y, 10.0, col.GSW_ZERO_10Y, col.GSW_ZERO_9Y),
    ZeroTenor(col.UST_ZERO_30Y, 30.0, col.GSW_ZERO_30Y, col.GSW_ZERO_29Y),
)


def _yield_value(row: pd.Series, name: str) -> float:
    value = row[name]
    if not isinstance(value, (float, np.floating)):
        raise TypeError(f"Expected numeric yield in {name}, got {type(value).__name__}")
    return float(value)


def _yield_at(row: pd.Series, tenor: ZeroTenor, years: float) -> float:
    lower = tenor.years - 1.0
    weight = (years - lower) / (tenor.years - lower)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Invalid maturity {years} for {tenor.name}")

    low = _yield_value(row, tenor.lower_yield)
    high = _yield_value(row, tenor.target_yield)
    return low * (1.0 - weight) + high * weight


def _price(row: pd.Series, tenor: ZeroTenor, years: float) -> float:
    rate = _yield_at(row, tenor, years) / 100.0
    return float(np.exp(-rate * years))


def _month_ends(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[frame[list(col.GSW_ZERO_YIELDS)].notna().all(axis=1)]
    index = utils.expect(valid.index, pd.DatetimeIndex)
    months = pd.Index(index.strftime("%Y-%m"), name="Month")
    monthly = valid.groupby(months).tail(1)
    monthly.index = pd.DatetimeIndex(
        utils.expect(monthly.index, pd.DatetimeIndex) + pd.offsets.MonthEnd(0),
        name="Month",
    )
    return monthly.sort_index()


def zero_returns(frame: pd.DataFrame) -> pd.DataFrame:
    monthly = _month_ends(frame)
    returns: dict[str, dict[pd.Timestamp, float]] = {
        tenor.name: {} for tenor in ZERO_TENORS
    }

    for previous, current in zip(monthly.index[:-1], monthly.index[1:]):
        # Elapsed days use month-end-normalized dates, not raw GSW observation dates.
        elapsed = (current - previous).days / DAYS_PER_YEAR
        for tenor in ZERO_TENORS:
            before = _price(monthly.loc[previous], tenor, tenor.years)
            after = _price(monthly.loc[current], tenor, tenor.years - elapsed)
            returns[tenor.name][current] = after / before - 1.0

    result = pd.DataFrame(returns)
    result.index.name = "Month"
    return result.sort_index()
