from ironbridge_trader.analytics import validation_bar


def test_thresholds_are_within_sane_ranges():
    # Guards against a typo (e.g. a p-value of 5 instead of 0.05) doing
    # real damage silently -- these values are a written commitment
    # (docs/validation-plan.md), so a bad constant here is worse than
    # a bad default elsewhere.
    assert validation_bar.MIN_SHARPE_RATIO_VS_BENCHMARK > 0
    assert validation_bar.MAX_DRAWDOWN_MULTIPLE_OF_BENCHMARK >= 1.0
    assert validation_bar.PERMUTATION_TEST_RUNS >= 30  # enough runs for the p-value to mean anything
    assert 0 < validation_bar.PERMUTATION_TEST_MAX_P_VALUE < 0.5
