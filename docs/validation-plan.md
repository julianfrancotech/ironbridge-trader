# Validation plan: the pre-committed bar for "this beats the benchmark"

**Written 2026-09-11, before any multi-regime backtest has been run
against it.** That ordering is the entire point of this document —
see "Why this is written down, and why it's not a Settings field"
below. If you're reading this after already having seen a backtest
result you didn't like, the discipline this document exists to enforce
has already failed; don't edit the thresholds below in response to
that result. Open an issue/PR instead, with a reason that has nothing
to do with the number you just saw.

## The benchmark

`analytics.performance.compute_buy_and_hold_curve` — an equal-dollar
allocation across the current watchlist, bought once on day one of the
evaluation window, held with zero decisions and zero trades after
that. This is the thing every active strategy in this app has to beat
to be worth its own complexity.

## The bar: all four criteria must hold, not just one

| # | Criterion | Threshold | Why this shape |
|---|---|---|---|
| A | **Net return** | Strategy's total return over the eval window ≥ benchmark's, both net of `Settings.transaction_cost_bps` + `commission_per_trade` | The obvious one — but alone it's not sufficient, hence B–D |
| B | **Risk-adjusted** | Strategy's Sharpe ratio ≥ benchmark's Sharpe ratio | Blocks "won on return only by taking on much more risk" |
| C | **Drawdown** | Strategy's max drawdown ≤ 1.25× the benchmark's max drawdown | Blocks "won on A and B, but with one drawdown month nobody would actually sit through live" |
| D | **Not luck** | Permutation test: shuffle the training labels and retrain 100 times; the real model's excess return over the benchmark must beat at least 95 of those 100 random-label runs (p < 0.05) | The check most people skip. A–C can all pass on one time window by chance alone — this is what actually distinguishes signal from noise |

All four are AND'd together. A strategy that wins on return and passes
the permutation test but only by taking on materially more risk or
drawdown than the benchmark does **not** clear this bar.

The named values behind each threshold (`1.0`, `1.25`, `100`, `0.05`)
live in code, not here in prose or in `config.py`'s `Settings` — see
"Why this is written down, and why it's not a Settings field" below
for exactly where and why.

## Where it gets measured

- A **sealed evaluation window** that is never touched during any
  iteration on features, hyperparameters, or the decision threshold —
  distinct from the walk-forward eval already used during normal
  model development (`orchestration/retrain_model.py`'s
  `eval_accuracy` / `eval_auc`). Checked exactly once, right before
  any real-money conversation.
- That window must span **at least one identifiable regime change** —
  a rally, a drawdown, a choppy/sideways stretch — not one continuous
  span that happens to be all one kind of market. See the multi-regime
  backtesting work this document's bar will eventually be checked
  against.
- The live paper-trading validation period (Phase 1) is measured
  against this same bar, over a window long enough to plausibly cross
  a real regime change — months, not weeks. The exact minimum length
  is a decision for when that phase starts, not this document.

## Why this is written down, and why it's not a Settings field

Every other tunable value in this app (`risk_fraction`,
`transaction_cost_bps`, `decision_confidence_floor`, ...) is a policy
dial: safe to change with zero friction from the dashboard's Settings
tab, taking effect on the next run, because there's no harm in
adjusting risk tolerance or a cost assumption whenever you want.

This bar is the opposite kind of thing. Its entire value is as a
commitment device against the single most common way people fool
themselves in this kind of work: deciding what "good enough" means
*after* seeing the number, so that whatever result showed up
retroactively looks like it clears the bar. Putting these thresholds
in `Settings` — editable from the same screen as `risk_fraction`,
taking effect immediately, no review — would make that failure mode
one slider-drag away.

Instead, the threshold values live as named constants in
`analytics/validation_bar.py`, imported by whatever eventually runs
the multi-regime backtest against them. They're still genuinely
changeable — this is software, not a contract — but changing them
means a deliberate code edit, committed with a stated reason, not a
JSON override file edited on a whim. The friction is the point.
