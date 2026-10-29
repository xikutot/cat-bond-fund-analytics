import numpy as np
import pandas as pd

from analysis.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

INPUT_FILE = RAW_DATA_DIR / "monthly.csv"


def parse_percent(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan

    text = str(value).strip()
    is_percent = text.endswith("%")
    number = float(text.removesuffix("%").replace(",", ""))
    return number / 100 if is_percent else number


def load_data(path=INPUT_FILE):
    raw = pd.read_csv(path, dtype=str)
    label = raw.columns[0]
    columns = raw.columns[1:]

    dates = pd.to_datetime(raw[label], format="%Y-%m-%d", errors="coerce")
    monthly = dates.notna()

    returns = raw.loc[monthly, columns].apply(
        lambda col: col.map(parse_percent)
    )
    returns.index = pd.DatetimeIndex(dates.loc[monthly], name="Month")
    returns = returns.sort_index()

    def get_fees(row_name):
        rows = raw.loc[raw[label].eq(row_name), columns]
        if rows.empty:
            raise ValueError(f"Missing metadata row: {row_name}")
        return rows.iloc[0].map(parse_percent).reindex(returns.columns)

    return (
        returns,
        get_fees("Management Fee"),
        get_fees("Norm Management Fee"),
    )


def normalize_returns(returns, actual_fees, target_fees):
    actual = actual_fees.reindex(returns.columns)
    target = target_fees.reindex(returns.columns)

    missing = actual.notna() ^ target.notna()
    if missing.any():
        raise ValueError(
            f"Incomplete fee data: {list(returns.columns[missing.to_numpy()])}"
        )

    adjustment = (actual - target).fillna(0) / 12
    return np.expm1(np.log1p(returns).add(adjustment, axis="columns"))


def remove_cash_influence(returns, cash_returns):
    cash = cash_returns.reindex(returns.index)
    return (1 + returns).div(1 + cash, axis="index") - 1


def save_data(data, filename):
    path = PROCESSED_DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index_label="Month", float_format="%.8f")
    return path


def prepare_returns(path=INPUT_FILE, save=True):
    returns, actual_fees, target_fees = load_data(path)
    normalized = normalize_returns(returns, actual_fees, target_fees)

    if "Cash" not in normalized.columns:
        raise ValueError("Missing 'Cash' column.")

    excess = remove_cash_influence(
        normalized.drop(columns="Cash"),
        normalized["Cash"],
    )

    if save:
        save_data(normalized, "cat_bond_returns_fee_normalized.csv")
        save_data(excess, "cat_bond_returns_cash_excess.csv")

    return normalized, excess
