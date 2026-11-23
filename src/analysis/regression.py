from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from statsmodels.regression.linear_model import RegressionResultsWrapper

from analysis import columns as col, utils
from analysis.data import factors as data_factors


type _Fit = tuple[pd.DataFrame, RegressionResultsWrapper]
type _Mask = pd.Index | NDArray[np.bool_]


@dataclass(frozen=True)
class ExposureStudy:
    funds: tuple[str, ...]
    sample: pd.DataFrame
    sd: pd.Series
    cat: pd.DataFrame
    control: pd.DataFrame

    def expanding(self, *, anchor: str, min_obs: int) -> pd.DataFrame:
        return expanding_exposures(
            self.sample,
            funds=self.funds,
            target=col.CAT_PC1_EXCESS,
            sd=self.sd,
            end=anchor,
            min_obs=min_obs,
        )

    def rolling(self, *, anchor: str, window: int) -> pd.DataFrame:
        return rolling_exposures(
            self.sample,
            funds=self.funds,
            target=col.CAT_PC1_EXCESS,
            sd=self.sd,
            end=anchor,
            window=window,
        )


def _through(frame: pd.DataFrame, end: str | pd.Timestamp | None) -> pd.DataFrame:
    data = frame.sort_index()
    if end is None:
        return data
    cutoff = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    return data.loc[data.index <= cutoff]


def factor_sds(
    frame: pd.DataFrame,
    factors: Sequence[str],
    *,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Factor volatilities on one common sample, used for all exposure scales."""
    utils.expect_columns(frame, list(factors))
    data = _through(frame, end)[list(factors)].dropna()
    sds = utils.expect(data.std(ddof=1), pd.Series)
    finite = utils.expect(np.isfinite(sds), pd.Series)
    invalid = utils.expect(sds[sds.isna() | ~finite | (sds <= 0)], pd.Series)
    if not invalid.empty:
        raise ValueError(f"No variation in factors: {list(invalid.index)}")
    return sds


def _fit(
    frame: pd.DataFrame,
    *,
    name: str,
    factors: Sequence[str],
    min_obs: int,
    maxlags: int,
) -> _Fit | None:
    columns = [name, *factors]
    sample = utils.expect(frame[columns].dropna(), pd.DataFrame)
    if len(sample) < max(min_obs, len(columns) + 1):
        return None

    result = sm.OLS(
        sample[name],
        sm.add_constant(sample[list(factors)]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return sample, result


def _exposure_rows(
    frame: pd.DataFrame,
    *,
    funds: Sequence[str],
    factors: Sequence[str],
    sd: pd.Series,
    cash: str,
    min_obs: int,
    maxlags: int,
) -> list[dict[str, object]]:
    all_factors = [cash, *factors]
    utils.expect_columns(frame, [*funds, *all_factors])
    missing_sd = [factor for factor in factors if factor not in sd.index]
    if missing_sd:
        raise ValueError(f"Missing factor SDs: {missing_sd}")

    rows: list[dict[str, object]] = []
    for fund in funds:
        fit = _fit(
            frame,
            name=fund,
            factors=all_factors,
            min_obs=min_obs,
            maxlags=maxlags,
        )
        if fit is None:
            continue
        _, result = fit

        cash_beta = utils.as_float(result.params[cash], label=f"Cash beta for {fund}")
        cash_p = utils.as_float(result.pvalues[cash], label=f"{col.CASH_P} for {fund}")
        observations = int(
            utils.as_float(result.nobs, label=f"{col.OBSERVATIONS} for {fund}")
        )
        r2 = utils.as_float(result.rsquared, label=f"{col.R2} for {fund}")

        for factor in factors:
            coefficient = utils.as_float(
                result.params[factor], label=f"Beta {factor} for {fund}"
            )
            factor_sd = utils.as_float(sd.loc[factor], label=f"SD {factor}")
            rows.append(
                {
                    col.FUND: fund,
                    col.FACTOR: factor,
                    col.COEFFICIENT: coefficient,
                    col.FACTOR_SD: factor_sd,
                    col.IMPACT_PER_SD: coefficient * factor_sd * 100,
                    col.T_STAT: utils.as_float(
                        result.tvalues[factor], label=f"t {factor} for {fund}"
                    ),
                    col.P_VALUE: utils.as_float(
                        result.pvalues[factor], label=f"p {factor} for {fund}"
                    ),
                    col.OBSERVATIONS: observations,
                    col.R2: r2,
                    col.CASH_BETA: cash_beta,
                    col.CASH_P: cash_p,
                }
            )

    return rows


def factor_exposures(
    frame: pd.DataFrame,
    *,
    funds: Sequence[str],
    factors: Sequence[str],
    sd: pd.Series,
    cash: str = col.CASH,
    min_obs: int = 60,
    maxlags: int = 3,
) -> pd.DataFrame:
    """Full-sample exposures using one fixed factor SD for every fund."""
    if cash in factors:
        raise ValueError(f"{cash!r} belongs in `cash`, not `factors`")
    rows = _exposure_rows(
        frame,
        funds=funds,
        factors=factors,
        sd=sd,
        cash=cash,
        min_obs=min_obs,
        maxlags=maxlags,
    )
    return pd.DataFrame(rows, columns=list(col.EXPOSURE_COLUMNS))


def representation_exposures(
    frame: pd.DataFrame,
    *,
    funds: Sequence[str],
    factors: Sequence[str],
    sd: pd.Series,
    cash: str = col.CASH,
    min_obs: int = 36,
    maxlags: int = 3,
) -> pd.DataFrame:
    """Separate simple regressions for alternative CAT/ILS representations."""
    frames = [
        factor_exposures(
            frame,
            funds=funds,
            factors=(factor,),
            sd=sd,
            cash=cash,
            min_obs=min_obs,
            maxlags=maxlags,
        )
        for factor in factors
    ]
    return pd.concat(frames, ignore_index=True)


def _window_rows(
    frame: pd.DataFrame,
    *,
    funds: Sequence[str],
    target: str,
    sd: pd.Series,
    cash: str,
    windows: Sequence[tuple[str, _Mask]],
    min_obs: int,
    maxlags: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, mask in windows:
        for row in _exposure_rows(
            frame.loc[mask],
            funds=funds,
            factors=(target,),
            sd=sd,
            cash=cash,
            min_obs=min_obs,
            maxlags=maxlags,
        ):
            row[col.FACTOR] = label
            rows.append(row)
    return pd.DataFrame(rows, columns=list(col.EXPOSURE_COLUMNS))


def expanding_exposures(
    frame: pd.DataFrame,
    *,
    funds: Sequence[str],
    target: str,
    sd: pd.Series,
    cash: str = col.CASH,
    end: str | pd.Timestamp | None = None,
    min_obs: int = 36,
    maxlags: int = 3,
) -> pd.DataFrame:
    """Exposures over windows expanding through each calendar year-end."""
    data = _through(frame, end)
    if data.empty:
        raise ValueError("No observations before the requested end date")

    index = utils.expect(data.index, pd.DatetimeIndex)
    anchor = utils.expect(index[-1], pd.Timestamp)
    years = sorted({int(stamp.year) for stamp in index})
    windows: list[tuple[str, _Mask]] = []
    for year in years:
        year_end = utils.expect(pd.Timestamp(year, 12, 31), pd.Timestamp)
        end_date = min(year_end, anchor)
        label = f"{year} YTD" if year == anchor.year else str(year)
        windows.append((label, index <= end_date))

    return _window_rows(
        data,
        funds=funds,
        target=target,
        sd=sd,
        cash=cash,
        windows=windows,
        min_obs=min_obs,
        maxlags=maxlags,
    )


def rolling_exposures(
    frame: pd.DataFrame,
    *,
    funds: Sequence[str],
    target: str,
    sd: pd.Series,
    cash: str = col.CASH,
    end: str | pd.Timestamp | None = None,
    window: int = 36,
    min_obs: int | None = None,
    maxlags: int = 3,
) -> pd.DataFrame:
    """Exposures over fixed-length windows ending at each calendar year-end."""
    data = _through(frame, end)
    if data.empty:
        raise ValueError("No observations before the requested end date")

    index = utils.expect(data.index, pd.DatetimeIndex)
    anchor = utils.expect(index[-1], pd.Timestamp)
    years = sorted({int(stamp.year) for stamp in index})
    windows: list[tuple[str, _Mask]] = []
    for year in years:
        year_end = utils.expect(pd.Timestamp(year, 12, 31), pd.Timestamp)
        end_date = min(year_end, anchor)
        start_date = end_date - pd.DateOffset(months=window)
        label = f"{year} YTD" if year == anchor.year else str(year)
        windows.append((label, (index > start_date) & (index <= end_date)))

    return _window_rows(
        data,
        funds=funds,
        target=target,
        sd=sd,
        cash=cash,
        windows=windows,
        min_obs=window if min_obs is None else min_obs,
        maxlags=maxlags,
    )


def exposures(
    panels: Mapping[str, pd.DataFrame],
    funds: Sequence[str],
    factor_columns: Mapping[str, str],
    representations: Sequence[str],
    *,
    anchor: str,
    min_months: int,
    control_min_months: int,
) -> ExposureStudy:
    factor_panel = data_factors.panel(panels, factor_columns)
    sd = factor_sds(
        factor_panel,
        tuple(str(name) for name in factor_panel if name != col.CASH),
        end=anchor,
    )
    sample = panels[col.TOTAL_RETURNS][list(funds)].join(factor_panel, how="inner")
    cat = representation_exposures(
        sample,
        funds=funds,
        factors=representations,
        sd=sd,
        min_obs=min_months,
    )
    control = factor_exposures(
        sample,
        funds=funds,
        factors=tuple(factor_columns),
        sd=sd,
        min_obs=control_min_months,
    )
    return ExposureStudy(tuple(funds), sample, sd, cat, control)


def exposure_table(
    exposures: pd.DataFrame,
    values: Sequence[str],
    *,
    decimals: int = 3,
    sort_by: str | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Return exposure values as a fund-by-factor presentation table."""
    table = utils.expect(
        exposures.pivot(index=col.FUND, columns=col.FACTOR, values=values[0]),
        pd.DataFrame,
    )
    for value in values[1:]:
        part = exposures.pivot(index=col.FUND, columns=col.FACTOR, values=value)
        table = table.join(part.add_suffix(f" {value}"))
    if sort_by is not None:
        table = table.sort_values(sort_by, ascending=ascending)
    return table.round(decimals)


def control_diagnostics(
    exposures: pd.DataFrame,
    *,
    decimals: int = 3,
    sort_by: str = col.R2,
) -> pd.DataFrame:
    """Return one row of control-regression diagnostics per fund."""
    data = exposures.drop_duplicates(col.FUND).set_index(col.FUND)
    selected = utils.expect(
        data.loc[:, list(col.CONTROL_DIAGNOSTICS)], pd.DataFrame
    )
    return selected.sort_values(sort_by, ascending=False).round(decimals)
