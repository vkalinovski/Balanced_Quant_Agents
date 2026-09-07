"""Run the full holdout experiment and export publication-ready artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.balanced_quantagents import (
    build_momentum_weights,
    equal_weight_returns,
    load_data,
    performance_metrics,
    run_portfolio,
    select_configuration,
    unscaled_strategy_returns,
    yearly_returns,
)


VALIDATION = ("2018-01-01", "2020-12-31")
HOLDOUT = ("2021-01-01", "2023-12-31")
TRANSACTION_COST = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def stress_audit(returns: pd.Series, diagnostics: pd.DataFrame) -> pd.Series:
    sample = returns.loc[HOLDOUT[0] : HOLDOUT[1]]
    monthly = (1.0 + sample).resample("ME").prod() - 1.0
    negative = sample < 0
    streaks = negative.groupby((negative != negative.shift()).cumsum()).sum()
    return pd.Series(
        {
            "Worst day": sample.min(),
            "Best day": sample.max(),
            "Monthly 5% quantile": monthly.quantile(0.05),
            "Worst month": monthly.min(),
            "Positive month share": (monthly > 0).mean(),
            "Daily VaR 95%": sample.quantile(0.05),
            "Daily CVaR 95%": sample[sample <= sample.quantile(0.05)].mean(),
            "Longest losing streak (days)": int(streaks.max()),
            "Average gross exposure": diagnostics.loc[HOLDOUT[0] : HOLDOUT[1], "gross_exposure"].mean(),
            "Maximum gross exposure": diagnostics.loc[HOLDOUT[0] : HOLDOUT[1], "gross_exposure"].max(),
            "Annual turnover": diagnostics.loc[HOLDOUT[0] : HOLDOUT[1], "turnover"].sum() / 3.0,
            "Cumulative cost / initial capital": diagnostics.loc[HOLDOUT[0] : HOLDOUT[1], "cost"].sum(),
        },
        name="Balanced strategy",
    )


def _equity_plot(series_map: dict[str, pd.Series], output: Path) -> None:
    colors = {
        "Balanced strategy": "#0057B8",
        "Unscaled selected signal": "#E76F51",
        "Equal Weight 30": "#2A9D8F",
        "QQQ benchmark": "#6C757D",
    }
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for name, series in series_map.items():
        sample = series.loc[HOLDOUT[0] : HOLDOUT[1]]
        equity = (1.0 + sample).cumprod()
        ax.plot(equity.index, equity, label=name, color=colors[name], linewidth=2.2 if name == "Balanced strategy" else 1.5)
    ax.axhline(1.0, color="#222222", linewidth=0.8, alpha=0.5)
    ax.set_title(
        "Locked holdout equity curves",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=28,
    )
    ax.text(0, 1.01, "2021-2023, net of 10 bps one-way turnover cost", transform=ax.transAxes, color="#555555")
    ax.set_ylabel("Growth of $1")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _risk_plot(returns: pd.Series, diagnostics: pd.DataFrame, max_gross: float, output: Path) -> None:
    sample = returns.loc[HOLDOUT[0] : HOLDOUT[1]]
    equity = (1.0 + sample).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    exposure = diagnostics.loc[HOLDOUT[0] : HOLDOUT[1], "gross_exposure"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    axes[0].fill_between(drawdown.index, drawdown * 100, 0, color="#C44536", alpha=0.78)
    axes[0].set_title("Risk behavior on the locked holdout", loc="left", fontsize=16, fontweight="bold")
    axes[0].set_ylabel("Drawdown, %")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(exposure.index, exposure, color="#0057B8", linewidth=1.35)
    axes[1].axhline(max_gross, color="#C44536", linestyle="--", linewidth=1.0, label=f"Hard cap: {max_gross:.2f}x")
    axes[1].fill_between(exposure.index, 0, exposure, color="#0057B8", alpha=0.14)
    axes[1].set_ylabel("Gross exposure")
    axes[1].set_xlabel("Date")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    started = time.perf_counter()

    close, returns, qqq = load_data(root / "data")
    signal, risk, leaderboard = select_configuration(
        close, returns, qqq, transaction_cost=TRANSACTION_COST
    )
    weights = build_momentum_weights(close, returns, signal)
    balanced = run_portfolio(weights, returns, qqq, risk, TRANSACTION_COST)
    balanced_returns = balanced["returns"]
    positions = balanced["positions"]
    diagnostics = balanced["diagnostics"]
    assert isinstance(balanced_returns, pd.Series)
    assert isinstance(positions, pd.DataFrame)
    assert isinstance(diagnostics, pd.DataFrame)

    qqq_returns = qqq.pct_change(fill_method=None).fillna(0.0)
    series_map = {
        "Balanced strategy": balanced_returns,
        "Unscaled selected signal": unscaled_strategy_returns(weights, returns, TRANSACTION_COST),
        "Equal Weight 30": equal_weight_returns(returns, TRANSACTION_COST),
        "QQQ benchmark": qqq_returns,
    }
    performance = pd.DataFrame(
        {name: performance_metrics(series, *HOLDOUT) for name, series in series_map.items()}
    ).T
    annual = yearly_returns(series_map)
    stress = stress_audit(balanced_returns, diagnostics).to_frame()

    article_context = pd.DataFrame(
        {
            "Paper QuantAgents": {
                "Total Return": 2.9955,
                "ARR": 0.5868,
                "Sharpe": 3.11,
                "Max Drawdown": 0.1686,
                "Daily Volatility": 0.0143,
            },
            "Balanced reproduction": {
                "Total Return": performance.loc["Balanced strategy", "Total Return"],
                "ARR": performance.loc["Balanced strategy", "ARR"],
                "Sharpe": performance.loc["Balanced strategy", "Sharpe"],
                "Max Drawdown": performance.loc["Balanced strategy", "Max Drawdown"],
                "Daily Volatility": balanced_returns.loc[HOLDOUT[0] : HOLDOUT[1]].std(),
            },
        }
    ).T

    leaderboard.to_csv(results_dir / "validation_leaderboard.csv", index=False)
    performance.to_csv(results_dir / "performance_comparison.csv")
    annual.to_csv(results_dir / "yearly_returns.csv")
    stress.to_csv(results_dir / "stress_audit.csv")
    article_context.to_csv(results_dir / "article_context.csv")
    positions.loc[HOLDOUT[0] : HOLDOUT[1]].to_csv(results_dir / "daily_positions.csv")
    diagnostics.loc[HOLDOUT[0] : HOLDOUT[1]].to_csv(results_dir / "daily_diagnostics.csv")

    _equity_plot(series_map, figures_dir / "equity_curves.png")
    _risk_plot(balanced_returns, diagnostics, risk.max_gross, figures_dir / "drawdown_and_exposure.png")

    manifest = {
        "status": "completed",
        "method": "balanced validation objective; no paper-metric calibration",
        "validation_period": list(VALIDATION),
        "locked_holdout_period": list(HOLDOUT),
        "transaction_cost_one_way": TRANSACTION_COST,
        "signal": asdict(signal),
        "risk": asdict(risk),
        "selection_score": leaderboard.iloc[0]["validation_score"],
        "holdout_result": performance.loc["Balanced strategy"].to_dict(),
        "data_sha256": {
            name: _sha256(root / "data" / name)
            for name in ["nasdaq30_2010_2023.csv", "market_factors.csv"]
        },
        "python": platform.python_version(),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "Fixed equity universe may contain survivorship bias.",
            "No market-impact or borrow model beyond turnover costs.",
            "A three-year holdout is informative but not definitive.",
        ],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_value), encoding="utf-8"
    )
    (results_dir / "selected_configuration.json").write_text(
        json.dumps({"signal": asdict(signal), "risk": asdict(risk)}, indent=2),
        encoding="utf-8",
    )
    return {
        "close": close,
        "returns": returns,
        "qqq": qqq,
        "signal": signal,
        "risk": risk,
        "leaderboard": leaderboard,
        "weights": weights,
        "balanced": balanced,
        "series_map": series_map,
        "performance": performance,
        "annual": annual,
        "stress": stress,
        "article_context": article_context,
        "manifest": manifest,
    }
