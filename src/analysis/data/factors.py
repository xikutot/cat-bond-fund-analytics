from collections.abc import Mapping

import numpy as np
import pandas as pd

from analysis import columns as col, utils


def cat_ils_pc1(
    excess_returns: pd.DataFrame,
    *,
    plenum: str,
    intelligence: str,
    name: str = col.CAT_PC1_EXCESS,
) -> pd.Series:
    """First principal component of two standardized CAT/ILS excess returns."""
    columns = [plenum, intelligence]
    data = utils.expect_columns(excess_returns, columns)[columns].dropna()
    values = data.to_numpy(dtype=float)
    scales = values.std(axis=0, ddof=1)

    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("CAT/ILS indexes need positive variance for PCA")

    standardized = (values - values.mean(axis=0)) / scales
    _, _, right = np.linalg.svd(standardized, full_matrices=False)
    loadings = right[0]
    if loadings.sum() < 0:
        loadings = -loadings

    score = standardized @ loadings
    return pd.Series(score * float(scales.mean()), index=data.index, name=name)


def panel(
    panels: Mapping[str, pd.DataFrame],
    factor_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Build the joined factor panel used by exposure regressions."""
    frame = panels[col.ASSET_CLASS_RETURNS][[col.CASH]].copy()
    frame[col.PLENUM_CAT_EXCESS] = panels[col.EXCESS_RETURNS][col.PLENUM_CAT_INDEX]
    frame[col.ILS_EXCESS] = panels[col.EXCESS_RETURNS][col.ILS_INDEX]
    frame[col.CAT_PC1_EXCESS] = cat_ils_pc1(
        panels[col.EXCESS_RETURNS],
        plenum=col.PLENUM_CAT_INDEX,
        intelligence=col.ILS_INDEX,
    )

    for label, source in factor_columns.items():
        if label not in frame:
            frame[label] = panels[col.EXCESS_RETURNS][source]

    return utils.expect(frame, pd.DataFrame)
