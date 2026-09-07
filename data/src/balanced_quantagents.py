"""Core portfolio logic for the balanced QuantAgents-style reproduction.

All model selection is performed before 2021. The 2021-2023 interval is a
locked holdout and is never referenced by ``select_configuration``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class SignalConfig:
    lookback: int
    skip: int
    top_n: int
    inverse_vol: bool


@dataclass(frozen=True)
class RiskConfig:
    target_vol: float
    max_gross: float
    risk_off_multiplier: float
    soft_drawdown: float
    hard_drawdown: float
    soft_multiplier: float
    hard_multiplier: float


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load adjusted closes, aligned returns, and the QQQ benchmark."""
    data_dir = Path(data_dir)
    market = pd.read_csv(data_dir / "nasdaq30_2010_2023.csv", parse_dates=["date"])
    close = (
        market.pivot(index="date", columns="ticker", values="close")
        .sort_index()
        .loc["2010":"2023"]
        .ffill(limit=2)
        .dropna()
    )
    returns = close.pct_change(fill_method=None).fillna(0.0)

    factors = pd.read_csv(data_dir / "market_factors.csv", index_col=0, parse_dates=True)
    qqq = factors["QQQ"].reindex(close.index).ffill().dropna()
    common_index = close.index.intersection(qqq.index)
    return close.loc[common_index], returns.loc[common_index], qqq.loc[common_index]


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last observed trading date in each calendar month."""
    dates = pd.Series(index, index=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).max().values)


def build_momentum_weights(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    config: SignalConfig,
) -> pd.DataFrame:
    """Create long-only monthly momentum weights using data available that day."""
    rows: list[pd.DataFrame] = []
    for date in month_end_dates(close.index):
        i = close.index.get_loc(date)
        if i < config.lookback + config.skip + 63:
            weight = pd.Series(1.0 / close.shape[1], index=close.columns)
        else:
            end_i = i - config.skip
            momentum = close.iloc[end_i] / close.iloc[end_i - config.lookback] - 1.0
            selected = momentum.nlargest(config.top_n).index
            weight = pd.Series(0.0, index=close.columns)
            if config.inverse_vol:
                trailing_vol = returns.iloc[i - 63 : i].std().replace(0.0, np.nan)
                weight.loc[selected] = 1.0 / trailing_vol.loc[selected]
            else:
                weight.loc[selected] = 1.0
            weight = weight.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            weight /= weight.sum()
        rows.append(pd.DataFrame([weight], index=[date]))

    return pd.concat(rows).reindex(close.index).ffill().fillna(0.0)


def qqq_risk_off_signal(qqq: pd.Series) -> pd.Series:
    """Causal market stress flag based on QQQ trend and medium-term momentum."""
    raw = (qqq < qqq.rolling(200).mean()) & (qqq.pct_change(60) < 0)
    return raw.shift(1, fill_value=False).astype(bool)


def run_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    qqq: pd.Series,
    risk: RiskConfig,
    transaction_cost: float = 0.001,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Apply causal volatility targeting, drawdown control, and trading costs."""
    # The one-day delay prevents same-close signal execution.
    base_positions = weights.shift(1).fillna(0.0)
    base_returns = (base_positions * returns).sum(axis=1)

    estimated_vol = base_returns.rolling(63).std().shift(1) * np.sqrt(TRADING_DAYS)
    vol_scale = (
        risk.target_vol / estimated_vol.replace(0.0, np.nan)
    ).clip(lower=0.35, upper=risk.max_gross).fillna(1.0)

    base_equity = (1.0 + base_returns).cumprod()
    prior_drawdown = (base_equity / base_equity.cummax() - 1.0).shift(1).fillna(0.0)
    drawdown_scale = pd.Series(1.0, index=weights.index)
    drawdown_scale.loc[prior_drawdown <= risk.soft_drawdown] = risk.soft_multiplier
    drawdown_scale.loc[prior_drawdown <= risk.hard_drawdown] = risk.hard_multiplier

    risk_off = qqq_risk_off_signal(qqq)
    regime_scale = pd.Series(
        np.where(risk_off, risk.risk_off_multiplier, 1.0),
        index=weights.index,
    )
    total_scale = (vol_scale * drawdown_scale * regime_scale).clip(upper=risk.max_gross)
    positions = base_positions.mul(total_scale, axis=0)

    gross_returns = (positions * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    costs = transaction_cost * turnover
    net_returns = gross_returns - costs

    diagnostics = pd.DataFrame(
        {
            "estimated_vol": estimated_vol,
            "vol_scale": vol_scale,
            "prior_base_drawdown": prior_drawdown,
            "drawdown_scale": drawdown_scale,
            "risk_off": risk_off.astype(int),
            "regime_scale": regime_scale,
            "gross_exposure": positions.abs().sum(axis=1),
            "turnover": turnover,
            "cost": costs,
            "gross_return": gross_returns,
            "net_return": net_returns,
        }
    )
    return {
        "returns": net_returns,
        "positions": positions,
        "turnover": turnover,
        "diagnostics": diagnostics,
    }


def performance_metrics(returns: pd.Series, start: str, end: str) -> dict[str, float]:
    """Calculate annualized risk and return statistics for an inclusive period."""
    sample = returns.loc[start:end].dropna()
    if sample.empty:
        raise ValueError(f"No observations between {start} and {end}")
    equity = (1.0 + sample).cumprod()
    years = len(sample) / TRADING_DAYS
    annual_return = equity.iloc[-1] ** (1.0 / years) - 1.0
    annual_vol = sample.std() * np.sqrt(TRADING_DAYS)
    max_drawdown = abs((equity / equity.cummax() - 1.0).min())
    sharpe = sample.mean() / sample.std() * np.sqrt(TRADING_DAYS)
    downside = np.sqrt(np.mean(np.minimum(sample, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    return {
        "Total Return": equity.iloc[-1] - 1.0,
        "ARR": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe": sharpe,
        "Sortino": sample.mean() * TRADING_DAYS / downside,
        "Max Drawdown": max_drawdown,
        "Calmar": annual_return / max_drawdown,
    }


def candidate_signal_configs() -> list[SignalConfig]:
    return [
        SignalConfig(*values)
        for values in product([60, 120], [0, 21], [5, 8, 10], [False, True])
    ]


def candidate_risk_configs() -> list[RiskConfig]:
    governors = [
        (-0.08, -0.15, 0.85, 0.65),
        (-0.10, -0.20, 0.85, 0.65),
        (-0.12, -0.20, 0.90, 0.70),
    ]
    return [
        RiskConfig(target, maximum, regime, *governor)
        for target, maximum, regime, governor in product(
            [0.26, 0.28, 0.30, 0.32],
            [1.25, 1.50],
            [0.75, 0.90, 1.00],
            governors,
        )
    ]


def _rolling_validation_metrics(returns: pd.Series) -> list[dict[str, float]]:
    return [performance_metrics(returns, str(year - 2), str(year)) for year in range(2016, 2021)]


def select_configuration(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    qqq: pd.Series,
    signal_configs: Iterable[SignalConfig] | None = None,
    risk_configs: Iterable[RiskConfig] | None = None,
    transaction_cost: float = 0.001,
) -> tuple[SignalConfig, RiskConfig, pd.DataFrame]:
    """Select a balanced configuration using only observations through 2020."""
    signal_configs = list(signal_configs or candidate_signal_configs())
    risk_configs = list(risk_configs or candidate_risk_configs())
    rows: list[dict[str, float | int | bool]] = []

    for signal in signal_configs:
        weights = build_momentum_weights(close, returns, signal)
        for risk in risk_configs:
            result = run_portfolio(weights, returns, qqq, risk, transaction_cost)
            net_returns = result["returns"]
            turnover = result["turnover"]
            assert isinstance(net_returns, pd.Series)
            assert isinstance(turnover, pd.Series)
            validation = performance_metrics(net_returns, "2018", "2020")
            rolling = _rolling_validation_metrics(net_returns)

            # These constraints and weights define "balanced" before the holdout opens.
            if (
                validation["Max Drawdown"] > 0.30
                or validation["Annual Volatility"] > 0.36
                or validation["ARR"] < 0.30
            ):
                continue
            median_rolling_sharpe = float(np.median([x["Sharpe"] for x in rolling]))
            worst_rolling_sharpe = float(min(x["Sharpe"] for x in rolling))
            median_rolling_arr = float(np.median([x["ARR"] for x in rolling]))
            annual_turnover = float(turnover.loc["2018":"2020"].sum() / 3.0)
            score = (
                0.30 * validation["Sharpe"]
                + 0.25 * validation["Calmar"]
                + 0.20 * median_rolling_sharpe
                + 0.15 * worst_rolling_sharpe
                + 0.10 * median_rolling_arr
                - 0.01 * annual_turnover
            )
            rows.append(
                {
                    **{f"signal_{k}": v for k, v in asdict(signal).items()},
                    **{f"risk_{k}": v for k, v in asdict(risk).items()},
                    "validation_score": score,
                    "validation_arr": validation["ARR"],
                    "validation_sharpe": validation["Sharpe"],
                    "validation_max_drawdown": validation["Max Drawdown"],
                    "validation_volatility": validation["Annual Volatility"],
                    "median_rolling_sharpe": median_rolling_sharpe,
                    "worst_rolling_sharpe": worst_rolling_sharpe,
                    "median_rolling_arr": median_rolling_arr,
                    "annual_turnover": annual_turnover,
                }
            )

    leaderboard = pd.DataFrame(rows).sort_values("validation_score", ascending=False).reset_index(drop=True)
    if leaderboard.empty:
        raise RuntimeError("No candidate passed the pre-specified validation constraints")
    winner = leaderboard.iloc[0]
    signal = SignalConfig(
        int(winner.signal_lookback),
        int(winner.signal_skip),
        int(winner.signal_top_n),
        bool(winner.signal_inverse_vol),
    )
    risk = RiskConfig(
        float(winner.risk_target_vol),
        float(winner.risk_max_gross),
        float(winner.risk_risk_off_multiplier),
        float(winner.risk_soft_drawdown),
        float(winner.risk_hard_drawdown),
        float(winner.risk_soft_multiplier),
        float(winner.risk_hard_multiplier),
    )
    return signal, risk, leaderboard


def equal_weight_returns(returns: pd.DataFrame, transaction_cost: float = 0.001) -> pd.Series:
    weights = pd.DataFrame(1.0 / returns.shape[1], index=returns.index, columns=returns.columns)
    positions = weights.shift(1).fillna(0.0)
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    return (positions * returns).sum(axis=1) - transaction_cost * turnover


def unscaled_strategy_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = 0.001,
) -> pd.Series:
    """Return the selected signal at 1.0x gross exposure, after costs."""
    positions = weights.shift(1).fillna(0.0)
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    return (positions * returns).sum(axis=1) - transaction_cost * turnover


def yearly_returns(series_map: dict[str, pd.Series], start: int = 2021, end: int = 2023) -> pd.DataFrame:
    rows = {}
    for name, series in series_map.items():
        rows[name] = {
            year: (1.0 + series.loc[str(year)]).prod() - 1.0
            for year in range(start, end + 1)
        }
    return pd.DataFrame(rows).rename_axis("Year")
