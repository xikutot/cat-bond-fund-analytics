import numpy as np
import pandas as pd

from analysis import columns as col, utils

MONTHLY_PRECISION_PP: float = 0.01
ANNUAL_PRECISION_PP: float = 0.01
TOL_PP: float = 12 * MONTHLY_PRECISION_PP / 2 + ANNUAL_PRECISION_PP / 2  # 0.065 pp


def annual_deviations(monthly: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    columns = yearly.columns.intersection(monthly.columns)
    years = utils.expect(monthly.index, pd.DatetimeIndex).year
    compounded = (1 + monthly[columns]).groupby(years).prod() - 1
    complete = monthly[columns].groupby(years).count().eq(12)

    reported = yearly[columns]
    compounded_years = compounded.reindex(reported.index)
    complete_years = complete.reindex(reported.index).fillna(False)
    deviations = compounded_years.sub(reported).where(complete_years).mul(100)
    return utils.expect(deviations, pd.DataFrame)


def summarize(
    deviations: pd.DataFrame, tol: float = TOL_PP
) -> tuple[pd.DataFrame, pd.Series]:
    absolute = deviations.abs()
    flags = absolute.stack().loc[lambda values: values > tol].sort_values(
        ascending=False
    )

    summary = pd.DataFrame(
        {
            col.CHECK_YEARS: deviations.notna().sum(),
            col.CHECK_MEAN: deviations.mean(),
            col.CHECK_MEDIAN: deviations.median(),
            col.CHECK_MEAN_ABS: absolute.mean(),
            col.CHECK_RMSE: np.sqrt(deviations.pow(2).mean()),
            col.CHECK_MIN: deviations.min(),
            col.CHECK_MAX: deviations.max(),
            col.CHECK_WORST_YEAR: absolute.idxmax(),
            col.CHECK_FLAGS: (absolute > tol).sum(),
        }
    )

    return summary.sort_values(col.CHECK_MEAN_ABS, ascending=False), flags
