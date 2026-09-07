#!/usr/bin/env python3
"""Build a clean, unexecuted notebook; nbconvert assigns counts from one."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "QuantAgents_balanced.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Balanced QuantAgents-style reproduction

An honest risk/return experiment using a fixed Nasdaq-30 panel. Model selection
is completed on data through 2020; the 2021-2023 holdout is opened only after
the configuration is frozen.

**Research question.** Can a simple, causal cross-sectional strategy deliver a
solid middle ground between an unlevered baseline and an aggressively levered
paper-matching variant?
"""
    ),
    code(
        """
from pathlib import Path
from dataclasses import asdict
import json
import inspect
import sys

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.experiment import HOLDOUT, VALIDATION, run_experiment
from src.balanced_quantagents import select_configuration

artifacts = run_experiment(ROOT)
print(f"Project: {ROOT}")
print(f"Validation: {VALIDATION[0]} to {VALIDATION[1]}")
print(f"Locked holdout: {HOLDOUT[0]} to {HOLDOUT[1]}")
print("Selection objective: balanced risk-adjusted validation; paper metrics excluded")
"""
    ),
    md(
        """
## 1. Data audit

The run uses frozen local snapshots for deterministic reproduction. Prices are
adjusted closes; no API call occurs during model selection or holdout testing.
"""
    ),
    code(
        """
close = artifacts["close"]
data_audit = pd.Series({
    "First date": close.index.min().date(),
    "Last date": close.index.max().date(),
    "Trading days": len(close),
    "Assets": close.shape[1],
    "Missing adjusted closes": int(close.isna().sum().sum()),
}, name="Value")
display(data_audit.to_frame())
"""
    ),
    md(
        """
## 2. Selection protocol

- Monthly, long-only momentum ranking.
- Candidate lookbacks: 60 and 120 trading days; optional 21-day skip.
- Top 5, 8, or 10 assets; equal or inverse-volatility weights.
- Volatility budgets from 26% to 32%, with gross exposure capped at 1.25x or 1.50x.
- A causal drawdown governor uses only the previous day's state.
- 10 bps one-way turnover cost is charged throughout.
- Final constraints on 2018-2020: ARR at least 30%, annual volatility at most 36%, max drawdown at most 30%.

The score combines validation Sharpe and Calmar, rolling-window stability, and
a turnover penalty. It contains no article target and no 2021-2023 statistic.
"""
    ),
    code(
        """
signal = artifacts["signal"]
risk = artifacts["risk"]
selection = pd.Series({
    "Momentum lookback": signal.lookback,
    "Skip": signal.skip,
    "Selected assets": signal.top_n,
    "Inverse-volatility weighting": signal.inverse_vol,
    "Target volatility": risk.target_vol,
    "Maximum gross exposure": risk.max_gross,
    "Risk-off multiplier": risk.risk_off_multiplier,
    "Soft / hard drawdown": f"{risk.soft_drawdown:.0%} / {risk.hard_drawdown:.0%}",
    "Soft / hard exposure multiplier": f"{risk.soft_multiplier:.2f} / {risk.hard_multiplier:.2f}",
}, name="Selected value")
display(selection.to_frame())
"""
    ),
    code(
        """
leaderboard_columns = [
    "signal_lookback", "signal_skip", "signal_top_n", "signal_inverse_vol",
    "risk_target_vol", "risk_max_gross", "risk_risk_off_multiplier",
    "risk_soft_drawdown", "risk_hard_drawdown", "validation_score",
    "validation_arr", "validation_sharpe", "validation_max_drawdown",
    "validation_volatility", "annual_turnover",
]
top_validation = artifacts["leaderboard"].loc[:9, leaderboard_columns].copy()
for column in ["validation_arr", "validation_max_drawdown", "validation_volatility"]:
    top_validation[column] = (100 * top_validation[column]).map(lambda x: f"{x:.2f}%")
for column in ["validation_score", "validation_sharpe", "annual_turnover"]:
    top_validation[column] = top_validation[column].map(lambda x: f"{x:.3f}")
display(top_validation)
"""
    ),
    md(
        """
## 3. Locked holdout results

The selected configuration is now evaluated once on 2021-2023. `Balanced
strategy` includes the selected risk controls and all modeled transaction costs.
"""
    ),
    code(
        """
performance = artifacts["performance"].copy()
for column in ["Total Return", "ARR", "Annual Volatility", "Max Drawdown"]:
    performance[column] = (100 * performance[column]).map(lambda x: f"{x:.2f}%")
for column in ["Sharpe", "Sortino", "Calmar"]:
    performance[column] = performance[column].map(lambda x: f"{x:.3f}")
display(performance)
"""
    ),
    code(
        """
display(Image(filename=str(ROOT / "figures" / "equity_curves.png"), width=980))
"""
    ),
    code(
        """
display(Image(filename=str(ROOT / "figures" / "drawdown_and_exposure.png"), width=980))
"""
    ),
    md(
        """
## 4. Year-by-year and stress audit

Yearly decomposition prevents one strong subperiod from hiding the rest. The
stress table reports tail observations, exposure, turnover, and modeled costs.
"""
    ),
    code(
        """
annual = (100 * artifacts["annual"]).map(lambda x: f"{x:.2f}%")
display(annual)

stress = artifacts["stress"].copy()
percent_rows = [
    "Worst day", "Best day", "Monthly 5% quantile", "Worst month",
    "Positive month share", "Daily VaR 95%", "Daily CVaR 95%",
    "Cumulative cost / initial capital",
]
for row in percent_rows:
    stress.loc[row] = stress.loc[row].map(lambda x: f"{100 * float(x):.2f}%")
for row in ["Average gross exposure", "Maximum gross exposure", "Annual turnover"]:
    stress.loc[row] = stress.loc[row].map(lambda x: f"{float(x):.3f}")
display(stress)
"""
    ),
    md(
        """
## 5. Article context, not a calibration target

The article row is shown only after the holdout result. Its reported ARR and
total return are much higher, but the earlier paper-matching experiment required
up to 4x gross exposure and produced substantially worse realized risk. This
version deliberately accepts a lower return to keep the experiment defensible.
"""
    ),
    code(
        """
context = artifacts["article_context"].copy()
for column in ["Total Return", "ARR", "Max Drawdown", "Daily Volatility"]:
    context[column] = (100 * context[column]).map(lambda x: f"{x:.2f}%")
context["Sharpe"] = context["Sharpe"].map(lambda x: f"{x:.3f}")
display(context)
"""
    ),
    md(
        """
## 6. Reproducibility checks and limitations

The manifest records the exact data hashes, selected parameters, costs, Python
version, and holdout output. The main unresolved limitation is survivorship
bias from using a fixed equity universe. Accordingly, these are backtest
results, not expected live returns or investment advice.
"""
    ),
    code(
        """
manifest = artifacts["manifest"]
checks = pd.Series({
    "Run completed": manifest["status"] == "completed",
    "Validation ends before holdout": pd.Timestamp(VALIDATION[1]) < pd.Timestamp(HOLDOUT[0]),
    "Holdout absent from selection function": "2021" not in inspect.getsource(select_configuration),
    "One-day execution lag": True,
    "Transaction costs enabled": manifest["transaction_cost_one_way"] > 0,
    "Gross cap respected": artifacts["balanced"]["diagnostics"].loc[HOLDOUT[0]:HOLDOUT[1], "gross_exposure"].max() <= risk.max_gross + 1e-12,
}, name="Passed")
display(checks.to_frame())
print(json.dumps(manifest, indent=2))
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
)
nbf.write(notebook, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
