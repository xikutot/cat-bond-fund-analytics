"""Raw, derived, and report labels shared across the analysis."""

# Raw metadata rows
MANAGEMENT_FEE: str = "Management Fee"
NORM_MANAGEMENT_FEE: str = "Norm Management Fee"

# Raw source columns
CASH: str = "Cash"
PLENUM_CAT_INDEX: str = "Plenum CAT Bond UCITS Fund Index - Master Capital"
ILS_INDEX: str = "With Intelligence ILS Index - USD Hedged"
MSCI_ACWI_INDEX: str = "MSCI ACWI Standard (Large+Mid Cap) - Gross"
SG_TREND_INDEX: str = "SG Trend Index"
SG_CTA_INDEX: str = "SG CTA Index"

GSW_DATE: str = "Date"
GSW_ZERO_2Y: str = "SVENY02"
GSW_ZERO_3Y: str = "SVENY03"
GSW_ZERO_4Y: str = "SVENY04"
GSW_ZERO_5Y: str = "SVENY05"
GSW_ZERO_9Y: str = "SVENY09"
GSW_ZERO_10Y: str = "SVENY10"
GSW_ZERO_29Y: str = "SVENY29"
GSW_ZERO_30Y: str = "SVENY30"
GSW_ZERO_YIELDS: tuple[str, ...] = (
    GSW_ZERO_2Y,
    GSW_ZERO_3Y,
    GSW_ZERO_4Y,
    GSW_ZERO_5Y,
    GSW_ZERO_9Y,
    GSW_ZERO_10Y,
    GSW_ZERO_29Y,
    GSW_ZERO_30Y,
)

# Constructed panel columns
UST_ZERO_3Y: str = "UST Zero 3Y Return"
UST_ZERO_5Y: str = "UST Zero 5Y Return"
UST_ZERO_10Y: str = "UST Zero 10Y Return"
UST_ZERO_30Y: str = "UST Zero 30Y Return"

# Panel keys
TREASURY_ZERO_RETURNS: str = "treasury_zero_returns"
ASSET_CLASS_RETURNS: str = "asset_class_returns"
FEE_NORMALIZED: str = "fee_normalized"
CASH_EXCESS: str = "cash_excess"
TOTAL_RETURNS: str = "total_returns"
EXCESS_RETURNS: str = "excess_returns"

# Regression factor labels
PLENUM_CAT_EXCESS: str = "Plenum CAT Bond Excess"
ILS_EXCESS: str = "With Intelligence ILS Excess"
CAT_PC1_EXCESS: str = "CAT/ILS PC1 Excess"
UST_3Y_EXCESS: str = "UST 3Y Excess"
UST_5Y_EXCESS: str = "UST 5Y Excess"
UST_10Y_EXCESS: str = "UST 10Y Excess"
UST_30Y_EXCESS: str = "UST 30Y Excess"
MSCI_EXCESS: str = "MSCI ACWI Excess"
SG_TREND_EXCESS: str = "SG Trend Excess"
SG_CTA_EXCESS: str = "SG CTA Excess"

# Exposure report fields
FUND: str = "Fund"
FACTOR: str = "Factor"
COEFFICIENT: str = "Coefficient"
FACTOR_SD: str = "Factor SD"
IMPACT_PER_SD: str = "Impact per SD (pp)"
T_STAT: str = "t"
P_VALUE: str = "p"
OBSERVATIONS: str = "Observations"
R2: str = "R2"
CASH_BETA: str = "Cash Beta"
CASH_P: str = "Cash p"
EXPOSURE_COLUMNS: tuple[str, ...] = (
    FUND,
    FACTOR,
    COEFFICIENT,
    FACTOR_SD,
    IMPACT_PER_SD,
    T_STAT,
    P_VALUE,
    OBSERVATIONS,
    R2,
    CASH_BETA,
    CASH_P,
)
CONTROL_DIAGNOSTICS: tuple[str, ...] = (CASH_BETA, CASH_P, OBSERVATIONS, R2)

# Scorecard report labels
MONTHS: str = "Months"
NET_EXCESS: str = "Net excess (pp)"
FINANCED_EXCESS: str = "Financed excess (pp)"
COV_PROXY: str = "Cov proxy (pp)"
MAX_DRAWDOWN: str = "Max DD (pp)"
CAT_BETA: str = "Cat beta"
CAT_T: str = "Cat t"
PLENUM_BETA: str = "Plenum beta"
ILS_BETA: str = "ILS beta"
MSCI_T: str = "MSCI t"
CTA_T: str = "CTA t"
DECISION: str = "Decision"
REASON: str = "Reason"
PASS: str = "Pass"
REJECT: str = "Reject"

# Raw check report labels
CHECK_YEARS: str = "years"
CHECK_MEAN: str = "mean"
CHECK_MEDIAN: str = "median"
CHECK_MEAN_ABS: str = "mean_abs"
CHECK_RMSE: str = "rmse"
CHECK_MIN: str = "min"
CHECK_MAX: str = "max"
CHECK_WORST_YEAR: str = "worst_year"
CHECK_FLAGS: str = "flags"
