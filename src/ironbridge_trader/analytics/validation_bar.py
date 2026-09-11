"""The pre-committed bar for "this strategy beats its benchmark" --
see docs/validation-plan.md for the full reasoning behind each
threshold below and why they live here rather than config.py's
Settings.

Deliberately NOT a Settings field / dashboard toggle. Settings exists
for policy dials that are meant to be changed with zero friction and
take effect on the next run (risk_fraction, transaction_cost_bps, ...).
This is the opposite: a commitment device against the single most
common self-deception in this kind of work -- deciding the bar
*after* seeing the number. Changing a threshold here means a
deliberate code edit, committed with a stated reason, not a silent
JSON override -- that friction is the point, not an oversight.
"""

from __future__ import annotations

# Strategy Sharpe ratio must be >= benchmark Sharpe ratio * this. 1.0
# means "at least as good," not better by some margin -- criterion A
# (net return) and D (the permutation test) are where the actual bar
# for "worth it" lives; this one exists to rule out "won on return only
# by taking on much more risk."
MIN_SHARPE_RATIO_VS_BENCHMARK = 1.0

# Strategy's max drawdown must not exceed the benchmark's by more than
# this multiple. 1.25 allows some extra drawdown for an active strategy
# (it's taking positions the benchmark doesn't), but not materially more.
MAX_DRAWDOWN_MULTIPLE_OF_BENCHMARK = 1.25

# How many label-shuffled retrainings the permutation test runs to
# build its random-chance baseline.
PERMUTATION_TEST_RUNS = 100

# The real model's excess return over the benchmark must beat at least
# this fraction of the permutation-test runs (p < 0.05, i.e. beat at
# least 95 of 100) for the result to count as signal rather than noise.
PERMUTATION_TEST_MAX_P_VALUE = 0.05
