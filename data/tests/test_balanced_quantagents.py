import unittest

import numpy as np
import pandas as pd

from src.balanced_quantagents import (
    RiskConfig,
    SignalConfig,
    build_momentum_weights,
    performance_metrics,
    run_portfolio,
)


class BalancedBacktestTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.index = pd.bdate_range("2018-01-01", periods=900)
        shocks = rng.normal(0.0004, 0.012, size=(len(self.index), 6))
        self.close = pd.DataFrame(
            100 * np.cumprod(1 + shocks, axis=0),
            index=self.index,
            columns=list("ABCDEF"),
        )
        self.returns = self.close.pct_change(fill_method=None).fillna(0.0)
        self.qqq = pd.Series(100 * np.cumprod(1 + shocks[:, 0]), index=self.index)

    def test_weights_are_long_only_and_fully_invested(self):
        config = SignalConfig(60, 0, 3, False)
        weights = build_momentum_weights(self.close, self.returns, config)
        self.assertTrue((weights >= 0).all().all())
        invested = weights.sum(axis=1) > 0
        np.testing.assert_allclose(weights.loc[invested].sum(axis=1), 1.0)

    def test_execution_uses_one_day_delay(self):
        config = SignalConfig(60, 0, 3, False)
        weights = build_momentum_weights(self.close, self.returns, config)
        risk = RiskConfig(0.28, 1.25, 1.0, -0.08, -0.15, 0.85, 0.65)
        result = run_portfolio(weights, self.returns, self.qqq, risk)
        positions = result["positions"]
        first_day = positions.index[0]
        self.assertEqual(float(positions.loc[first_day].sum()), 0.0)

    def test_gross_exposure_respects_cap(self):
        config = SignalConfig(60, 0, 3, False)
        weights = build_momentum_weights(self.close, self.returns, config)
        risk = RiskConfig(0.32, 1.25, 0.9, -0.08, -0.15, 0.85, 0.65)
        result = run_portfolio(weights, self.returns, self.qqq, risk)
        exposure = result["diagnostics"]["gross_exposure"]
        self.assertLessEqual(float(exposure.max()), 1.25 + 1e-12)

    def test_metrics_are_finite(self):
        metrics = performance_metrics(self.returns.mean(axis=1), "2018", "2020")
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
