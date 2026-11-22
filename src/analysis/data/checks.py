import numpy as np
import pandas as pd

from analysis.utils import expect

MONTHLY_PRECISION_PP: float = 0.01
ANNUAL_PRECISION_PP: float = 0.01
TOL_PP: float = 12 * MONTHLY_PRECISION_PP / 2 + ANNUAL_PRECISION_PP / 2  # 0.065 pp


def annual_deviations(monthly: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    columns = yearly.columns.intersection(monthly.columns)
    years = expect(monthly.index, pd.DatetimeIndex).year
    compounded = (1 + monthly[columns]).groupby(years).prod() - 1
    complete = monthly[columns].groupby(years).count().eq(12)

    reported = yearly[columns]
    compounded_years = compounded.reindex(reported.index)
    complete_years = complete.reindex(reported.index).fillna(False)
    deviations = compounded_years.sub(reported).where(complete_years).mul(100)
    return expect(deviations, pd.DataFrame)


def summarize(
    deviations: pd.DataFrame, tol: float = TOL_PP
) -> tuple[pd.DataFrame, pd.Series]:
    absolute = deviations.abs()
    flags = absolute.stack().loc[lambda values: values > tol].sort_values(
        ascending=False
    )

    summary = pd.DataFrame(
        {
            "years": deviations.notna().sum(),
            "mean": deviations.mean(),
            "median": deviations.median(),
            "mean_abs": absolute.mean(),
            "rmse": np.sqrt(deviations.pow(2).mean()),
            "min": deviations.min(),
            "max": deviations.max(),
            "worst_year": absolute.idxmax(),
            "flags": (absolute > tol).sum(),
        }
    )

    return summary.sort_values("mean_abs", ascending=False), flags
