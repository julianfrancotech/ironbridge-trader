# Ironbridge Trader

A paper-trading assistant that predicts short-term price direction for a
small watchlist and decides whether to buy, sell, or hold — with a
dashboard that shows *why* every decision was made.

**No real broker, no real money.** Every fill is simulated. See
[Trading theory](#trading-theory) and [Agentic AI theory](#agentic-ai-theory)
for the reasoning behind how it's built, and
[docs/platform-boundaries.md](docs/platform-boundaries.md) for what it
deliberately does and doesn't know about.

## Quick start

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e ".[dev]"

./.venv/bin/python scripts/fetch_data.py      # pull OHLCV history (yfinance, free)
./.venv/bin/python scripts/train_model.py     # train the quant model, print held-out accuracy
./.venv/bin/python scripts/run_paper_trader.py  # one live decision cycle across the watchlist
./.venv/bin/python scripts/backtest.py        # replay history, populate the dashboard's Backtest tab

./.venv/bin/streamlit run src/ironbridge_trader/dashboard/app.py
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to enable the
Claude tool-using agent. Without it, everything still runs end-to-end —
`run_paper_trader.py` falls back to a deterministic threshold rule.

## Architecture

```
Bar history (SQLite)
        │
        ▼
  features/engineering.py       -- technical indicators, normalized
        │
        ▼
  ml/model.py (PricePredictor)   -- prob_up: P(next bar closes higher)
        │
        ▼
  protocols.DecisionMaker  ──┬── services/trading_agent.py     (ClaudeTradingAgent)
        │                    └── services/threshold_agent.py   (ThresholdDecisionService, fallback)
        ▼
      Decision  (action, confidence, rationale)
        │
        ▼
  risk/manager.py   -- deterministic veto: position limits, margin
        │  (only if it passes)
        ▼
  adapters/paper_broker.py   -- simulated fill
        │
        ▼
  storage/db.py   -- every bar, prediction, decision, fill, equity point
        │
        ▼
  dashboard/app.py   -- Streamlit, reads the database, nothing else
```

`engine/trading_engine.py` (the per-symbol predict → decide → risk-check
→ execute loop) is pure dependency injection over four protocols
(`MarketDataSource`, `Predictor`, `DecisionMaker`, `ExecutionClient`) —
it never imports a concrete implementation. `orchestration/*.py` is
where those concrete implementations get chosen and wired together for
a real run (e.g. `orchestration/run_decision_cycle.py` decides Claude vs.
the threshold fallback based on whether `ANTHROPIC_API_KEY` is set), and
`scripts/*.py` are the thin CLI entrypoints that call into orchestration.
That three-way split — engine (pure logic) / orchestration (wiring) /
scripts (CLI) — means the orchestration functions are directly
importable and testable without going through argv, and is exactly the
seam a future scheduler or API layer would call into instead of
shelling out to scripts.

`RiskManager` and `Database` are not protocols; they're the two fixed
concerns everything else is built around, with exactly one
implementation each and no planned second one.

Every module's docstring explains the *why* behind that module — read
them if you want the long version of anything below.

## Correctness & resilience

Three things a scheduled job running unattended actually needs, added
deliberately rather than assumed:

- **Idempotent decision cycles.** `engine/trading_engine.py` checks
  `db.has_decision_for(symbol, latest_bar_time)` before doing any work.
  A retried or duplicated trigger — the realistic failure mode for a
  cron job — is a silent no-op: no second (possibly LLM-backed, possibly
  costly) decision, no duplicate row. `storage/db.py`'s
  `UNIQUE(symbol, ts)` constraint on `decisions` backstops this at the
  database level in case anything ever calls `insert_decision` without
  going through the check.
- **Retry with backoff on both external calls.** `adapters/market_data.py`
  (yfinance) and `services/trading_agent.py` (the Claude API) each wrap
  only their actual network call — not the whole function — in
  `resilience.retry_with_backoff`, and each names only the exception
  types that are genuinely transient (a dropped connection, a timeout, a
  rate limit, the provider's own 5xx). A permanent failure — a symbol
  that doesn't exist, a bad API key — fails once, immediately, instead
  of being retried into a slower, equally-doomed failure.
- **WAL mode + a connection timeout on every SQLite connection**
  (`storage/db.py`). Without this, a decision cycle writing while the
  dashboard reads is one unlucky timing away from "database is locked."
  This is the actual ceiling of what SQLite can safely do here — a
  process that writes far more often than once a day, or more than one
  writer at a time, would need Postgres (see platform-boundaries.md).

## Configuring it: the dashboard's Settings tab

Every risk and model-training knob in `config.py` is editable from the
dashboard's **⚙️ Settings** tab, without touching code — and every field
links to the real concept behind it (Kelly criterion, stop orders,
margin, walk-forward optimization, the efficient-market hypothesis,
Anthropic's own tool-use docs), each one an actual URL that was looked
up and confirmed to exist before being put in the UI, not guessed.

Two design choices worth knowing about:

- **Secrets stay out of it.** `ANTHROPIC_API_KEY` is never shown or
  editable here — only a "configured / not set" status badge. Settings
  writes only to `data/settings_overrides.json` (non-secret policy
  values), never to `.env`, specifically so this page is safe to leave
  open on a shared screen. `bar_interval` is deliberately not exposed
  either — it's an architectural assumption baked into
  `MIN_BARS_REQUIRED` and the walk-forward split, not a casual dial.
- **Not every save takes effect immediately.** `config.py::load_settings()`
  is called fresh on every Streamlit rerun, so the dashboard itself
  picks up a save right away. But `symbols`, `history_years`, and
  `feature_lookback` / `train_test_split_ratio` only actually change
  behavior the next time you run `fetch_data.py` / `train_model.py` —
  each script is its own process and reads the override file at
  startup. The Settings tab says this next to each field that needs it.

## Trading theory

**Bars, not ticks.** This app decides once a day, so a bar (one OHLCV
row) is the natural unit — it has an open/high/low/close/volume a tick
doesn't, which is what every indicator below is computed from. Free
intraday data from Yahoo Finance is also capped at 7–60 days of history,
nowhere near enough to train on; daily bars give years.

**Features are ratios, never raw prices** (`features/engineering.py`).
AAPL at $180 today says nothing about AAPL at $50 five years ago, but the
*shape* of recent action — is the short average above the long one, how
volatile has it been, is volume unusual — repeats across price levels and
across symbols. That's what lets one model, trained across the whole
watchlist and years of history, generalize instead of memorizing price
levels it will never see again.

**Gradient-boosted trees, not a neural net** (`ml/model.py`). This is a
few thousand rows and nine features — tabular data at a scale where
boosted trees reliably match or beat deep learning, need no GPU or
feature scaling, and stay inspectable. A neural net here would be
complexity with no corresponding benefit.

**A walk-forward split, always** (`ml/training.py`). Training rows are
strictly *before* evaluation rows in time — the model is never evaluated
on data that would have required seeing the future to train on. This
project's actual result, from the included 5-year default watchlist:

```
accuracy=0.491  auc=0.494
```

Essentially a coin flip. That's not a bug — it's the honest answer, and
it's consistent with decades of empirical finance (the weak-form
efficient-market hypothesis): next-day direction, predicted from price
and volume history alone, is very close to unpredictable, because if it
weren't, the predictability would already be arbitraged away. **This
project doesn't hide that number** — it's what `scripts/train_model.py`
prints and what the dashboard's Model tab charts. Treat `prob_up` as a
weak, noisy tilt worth *combining with other judgment* (which is exactly
what the agent layer below is for), never as a reliable forecast on its
own.

**A deterministic risk gate, applied after every decision, by anyone**
(`risk/manager.py`). Position-size and margin limits are checked in code,
not asked of the model or the agent — neither has the authority to skip
this. It's the one hard veto point in the whole pipeline, and both the
Claude agent and the fallback rule pass through the exact same gate.

**Position size is derived from risk, not fixed** (`risk/position_sizer.py`).
`size = (equity × risk_fraction) / (reference_price × stop_loss_fraction)`
— fixed-fractional sizing, so the same account risk tolerance produces a
*different* quantity for a $10 stock than for $110,000 of Bitcoin,
instead of the same flat number regardless of price. It deliberately
ignores the decision's own confidence score: that confidence comes from
a model whose overall walk-forward accuracy is close to a coin flip, so
scaling size directly off it would be the full-Kelly mistake — betting
big on a number that isn't validated enough to trust that much. When
even one whole unit costs more than the risk budget allows (a real,
observed outcome for BTC-USD at higher prices against this app's
default $100k account and 1% risk fraction), size() returns 0 and the
engine skips the trade rather than rounding up to 1 and silently taking
on more risk than asked for.

**The dashboard has to show whether any of this was worth it, not just
that trades happened.** The Portfolio tab computes the actual
account-wide equity curve — cash plus mark-to-market value across every
symbol (`analytics/performance.py`) — against a zero-decisions,
fully-invested-on-day-one buy-and-hold benchmark of the same watchlist,
plus max drawdown and Sharpe ratio. Buy-and-hold is the bar an active
strategy has to clear, after every decision and every trade, to be
worth the complexity; on this app's own backtest, it currently isn't
close — buy-and-hold outperforms the strategy by a wide margin, mostly
because the risk-sized position sizer correctly refuses to buy BTC-USD
at most of the prices it reached over the period. That's not a bug in
the sizer — it's the honest consequence of a fixed dollar risk budget
against a highly-priced, highly-volatile asset, and the dashboard is
built to surface it rather than hide it.

**The backtest is honestly caveated, not just honestly labeled.**
`orchestration/run_backtest.py` replays history bar-by-bar without
letting the model see future *bars* (see `adapters/replay_market_data.py`).
But it evaluates every simulated day with the *current* trained model,
which was itself trained on the full stored history — including data
from after many of the days being replayed. That means the backtest's
results are optimistic and are a way to *see decision flow*, not a
performance claim. The honest number is `ml/training.py`'s held-out
`eval_accuracy` / `eval_auc`, shown in both the training script's output
and the dashboard's Model tab.

## Agentic AI theory

**What "agentic" means here, specifically.** The Claude agent
(`services/trading_agent.py`) isn't one classification prompt — it drives
a bounded [ReAct-style](https://arxiv.org/abs/2210.03629) loop: given a
symbol, it chooses which of five tools to call (`get_quant_signal`,
`get_price_history`, `get_position`, `get_recent_decisions`,
`get_risk_limits`), in what order, and how many, before calling a sixth,
`submit_decision`, to end its turn. That's real autonomy over *how to
gather context* — a strongly directional `prob_up` might get a decision
after one tool call, a borderline one might get three. But it is
deliberately **not** autonomy over the consequential action: placing a
trade always still goes through the deterministic risk manager
afterward, whether the agent produced the decision or not. This is the
same "agent proposes, code disposes" boundary a production trading
platform draws between an exchange connection's stateful order routing
and its own deterministic accounting logic — an LLM adds exactly the
same kind of judgment risk a live exchange connection does, and gets
the same kind of veto layer in front of it.

**Structured output over prompt-and-parse.** `submit_decision` is a tool
with a JSON schema (`action`, `confidence`, `rationale`), not a
free-text response the code then tries to regex out of prose. This is
the current, reliable way to get an LLM to hand back a value your code
can trust the shape of.

**Grounding against hallucination.** The system prompt explicitly
requires the agent to base every claim on a tool result it actually
retrieved, never invented numbers. Every tool result is real data
already computed by the quant/risk layers, not the agent's own
arithmetic — so if it says "prob_up is 0.71," that number came from
`ml/model.py`, not from the model guessing at plausible-sounding stats.

**A form of memory.** `get_recent_decisions` lets the agent see its own
last few calls and rationales for that symbol before deciding again —
simple episodic memory (a database query, not a vector store — there's
no unstructured text to retrieve-and-rank here, so RAG machinery would
be pure overhead). It's there so the agent can notice it's been
flip-flopping on noise and choose to hold instead.

**Guardrails against the two standard agent-loop failure modes:**
- *Runaway tool loops*: `agent_max_tool_iterations` (default 6) bounds
  how many rounds the agent gets. Hit the limit without a
  `submit_decision` call, and `decide()` returns HOLD with an explicit
  "budget exhausted" rationale — never an exception, never a stuck
  process.
- *Unchecked consequential action*: covered above — the agent proposes,
  the risk manager and broker are the only things that can actually move
  a simulated position, enforced in code (`engine/trading_engine.py`),
  not by asking the agent nicely.

**The fallback isn't a placeholder — it's a control.**
`services/threshold_agent.py` is a deterministic rule (`abs(prob_up -
0.5) * 2` as confidence, thresholded) that satisfies the exact same
`DecisionMaker` protocol as the Claude agent. It exists for two reasons:
the app needs to run with zero API key, and — more interestingly — it
gives you something to compare the LLM agent against. Both write their
decisions to the same `decisions` table with an `agent_kind` column, so
the dashboard can show, over time, whether the tool-using agent's extra
reasoning actually changes outcomes versus "just trust the model above a
confidence floor." Not asserting the answer here is deliberate; it's a
question the dashboard is built to let you answer for yourself.

**Where retraining fits.** The model "learns over time" through
`scripts/train_model.py` (→ `orchestration/retrain_model.py`), which you
re-run (by hand, or on a cron) as new bars accumulate. Each run produces
a new timestamped version, logged with its held-out accuracy, and the
dashboard's Model tab charts accuracy across every version you've
trained. That's a deliberate choice over continuous/online learning
inside the live trading loop: an explicit, versioned, auditable retrain
step means one noisy day can never silently warp what the model
believes.

## Repository layout

```
src/ironbridge_trader/
  domain/                Bar, Order, Fill, Position, Decision, QuantSignal -- the data model
  protocols.py             the four swappable interfaces
  resilience.py             retry-with-backoff for external calls (yfinance, Claude)
  adapters/                 exchange/data-connection boundary: yfinance ingestion, backtest replay, paper broker
  services/                  external integrations: the Claude tool-using agent + its tools, the threshold fallback
  features/                   technical indicators
  ml/                          the predictive model: train, evaluate, version, predict
  risk/                         the deterministic pre-trade risk gate + fixed-fractional position sizing
  engine/                        the per-symbol predict/decide/execute loop (pure logic, DI'd collaborators)
  orchestration/                  workflow wiring: which adapters/services to use for a given run
  storage/                         SQLite persistence -- the single source of truth the dashboard reads
  analytics/                        account equity curve, buy-and-hold benchmark, drawdown, Sharpe
  dashboard/                        the Streamlit app
scripts/                 thin CLI entrypoints into orchestration/
docs/
  platform-boundaries.md  what this app does and deliberately doesn't know about, and what would need to
                           change for it to go from daily-batch to live/intraday
tests/                   unit + integration tests (pytest)
```

## Limitations

- Paper trading only. Wiring in a real broker means writing one class
  satisfying `protocols.ExecutionClient`; nothing else changes — but
  that's a deliberately unfinished extension point, not a hidden feature.
- Daily bars, not intraday. Fine for a "decide once a day" cadence, not
  built for anything faster — see platform-boundaries.md for what that
  transition would actually require.
- Predictive power is genuinely weak (see Trading theory above) — this
  is a demonstration of a clean, well-instrumented decision pipeline,
  not a claim that it beats the market.
- The backtest reuses one full-history-trained model across the whole
  replay window (see caveat above); it's for exploring decision flow,
  not for reporting a Sharpe ratio.
