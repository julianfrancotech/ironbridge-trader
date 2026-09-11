"""Streamlit dashboard: the window onto everything the app has done.

One file, reading straight out of storage/db.py -- no separate API
server, no build step (see docs/platform-boundaries.md for why this app
does not have an HTTP layer between the dashboard and the engine, unlike
a larger production trading service). Every panel maps to one thing the
project brief asked to be able to "understand": what the market did, what the
model predicted, what the decision-maker decided and *why* (its own
words, verbatim, never re-summarized), and how well the model's
accuracy has moved as it's retrained.

Run with:  streamlit run src/ironbridge_trader/dashboard/app.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ironbridge_trader.analytics import performance
from ironbridge_trader.config import DECIMAL_FIELDS, load_settings, save_overrides
from ironbridge_trader.storage.db import Database

st.set_page_config(page_title="Ironbridge Trader", layout="wide")

BACKTEST_DB_NAME = "backtest.db"

# Schema for the Settings tab: one dict per TUNABLE_FIELDS entry in
# config.py. "learn_more" is always a URL that was actually looked up
# and confirmed to exist, not guessed -- see README for the rationale
# on why that matters here specifically.
SETTINGS_FIELDS = [
    dict(
        key="trading_enabled", section="Trading control", kind="toggle",
        label="Trading enabled",
        help=(
            "The kill switch. Off skips every live decision cycle entirely (run_paper_trader.py "
            "logs and exits, no decisions, no orders) until turned back on -- independent of, and "
            "faster than, any code change. Does NOT affect scripts/backtest.py or scripts/train_model.py."
        ),
    ),
    dict(
        key="symbols", section="Watchlist & data", kind="tags",
        label="Watchlist (comma-separated symbols)",
        help=(
            "Symbols pulled by scripts/fetch_data.py and evaluated by every decision cycle. "
            "Changing this needs a re-run of fetch_data.py (and train_model.py, so the new "
            "symbol's history is part of training) before it does anything."
        ),
    ),
    dict(
        key="history_years", section="Watchlist & data", kind="number",
        label="History to fetch (years)", min_value=1, max_value=20, step=1,
        help="Years of daily bars scripts/fetch_data.py pulls per symbol. Takes effect on the next fetch_data.py run.",
    ),
    dict(
        key="max_concurrent_symbols", section="Watchlist & data", kind="number",
        label="Max concurrent symbols", min_value=1, max_value=50, step=1,
        help=(
            "How many symbols fetch_data.py and a decision cycle process in parallel. "
            "Higher is faster on a large watchlist but risks hitting the Anthropic API's "
            "or a broker's rate limits sooner."
        ),
        learn_more="https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor",
        learn_more_label="ThreadPoolExecutor (Python docs)",
    ),
    dict(
        key="data_provider", section="Watchlist & data", kind="select", options=["yfinance", "alpaca"],
        label="Market data source",
        help=(
            "yfinance needs no account. alpaca needs ALPACA_API_KEY/ALPACA_SECRET_KEY in .env "
            "(never entered here) -- falls back to yfinance with a logged warning if selected "
            "without them. Takes effect on the next fetch_data.py run."
        ),
        learn_more="https://docs.alpaca.markets/docs/about-market-data-api",
        learn_more_label="Market Data API (Alpaca docs)",
    ),
    dict(
        key="execution_provider", section="Watchlist & data", kind="select",
        options=["paper_broker", "alpaca_paper"],
        label="Execution provider",
        help=(
            "paper_broker fills every order instantly at its reference price, no account needed. "
            "alpaca_paper submits real orders to Alpaca's paper-trading endpoint (still no real "
            "money) and needs ALPACA_API_KEY/ALPACA_SECRET_KEY in .env -- falls back to "
            "paper_broker with a logged warning if selected without them."
        ),
        learn_more="https://docs.alpaca.markets/docs/paper-trading",
        learn_more_label="Paper Trading (Alpaca docs)",
    ),
    dict(
        key="feature_lookback", section="Model training", kind="number",
        label="Feature lookback (bars)", min_value=30, max_value=250, step=5,
        help="Bars of history the feature engineer needs before it can compute an indicator. Takes effect on the next train_model.py run.",
        learn_more="https://corporatefinanceinstitute.com/learn/resources/equities/moving-average",
        learn_more_label="Moving averages (Corporate Finance Institute)",
    ),
    dict(
        key="train_test_split_ratio", section="Model training", kind="slider",
        label="Train / eval split", min_value=0.5, max_value=0.95, step=0.05,
        help="Fraction of history used to train; the rest is held out, chronologically after it, for the honest accuracy number. Takes effect on the next train_model.py run.",
        learn_more="https://en.wikipedia.org/wiki/Walk_forward_optimization",
        learn_more_label="Walk-forward optimization (Wikipedia)",
    ),
    dict(
        key="decision_confidence_floor", section="Decision-making", kind="slider",
        label="Confidence floor", min_value=0.5, max_value=0.95, step=0.01,
        help="Below this confidence, the decision is HOLD. Raise it to trade less often, only on stronger signals. Takes effect on the next paper-trading cycle or backtest.",
        learn_more="https://en.wikipedia.org/wiki/Efficient-market_hypothesis",
        learn_more_label="Why to be skeptical of high confidence here: Efficient-market hypothesis (Wikipedia)",
    ),
    dict(
        key="agent_max_tool_iterations", section="Decision-making", kind="number",
        label="Agent tool-call budget", min_value=1, max_value=20, step=1,
        help="How many tool calls the Claude agent gets before it must decide. Hitting the limit defaults to HOLD, never an error.",
        learn_more="https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works",
        learn_more_label="How tool use works (Anthropic docs)",
    ),
    dict(
        key="anthropic_model", section="Decision-making", kind="text",
        label="Anthropic model ID",
        help="Model used by the Claude tool-using agent (services/trading_agent.py). Only used when ANTHROPIC_API_KEY is set in .env.",
        learn_more="https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works",
        learn_more_label="How tool use works (Anthropic docs)",
    ),
    dict(
        key="account_equity", section="Risk & position sizing", kind="number",
        label="Account equity ($)", min_value=1_000.0, max_value=10_000_000.0, step=1_000.0,
        help="The simulated account size every risk check and position-sizing calculation is measured against.",
    ),
    dict(
        key="max_position_size", section="Risk & position sizing", kind="number",
        label="Max position size (units)", min_value=1, max_value=100_000, step=1,
        help="Hard cap on units held in one symbol, enforced by risk/manager.py regardless of what the sizer computes.",
        learn_more="https://en.wikipedia.org/wiki/Risk_management",
        learn_more_label="Risk management (Wikipedia)",
    ),
    dict(
        key="margin_rate", section="Risk & position sizing", kind="slider",
        label="Margin rate", min_value=0.01, max_value=1.0, step=0.01,
        help="Fraction of an order's notional value held as margin, checked against account equity.",
        learn_more="https://www.finra.org/rules-guidance/key-topics/margin-accounts",
        learn_more_label="Margin Regulation (FINRA)",
    ),
    dict(
        key="risk_fraction", section="Risk & position sizing", kind="slider",
        label="Risk fraction per trade", min_value=0.0005, max_value=0.10, step=0.0005,
        help="Fraction of account equity you're willing to lose on one trade. Drives risk/position_sizer.py — higher means bigger positions for the same signal.",
        learn_more="https://en.wikipedia.org/wiki/Kelly_criterion",
        learn_more_label="Kelly criterion (Wikipedia)",
    ),
    dict(
        key="stop_loss_fraction", section="Risk & position sizing", kind="slider",
        label="Stop-loss distance (sizing only)", min_value=0.01, max_value=0.50, step=0.01,
        help="Assumed adverse-move distance used ONLY to size positions — this app does not place a real stop order. See risk/position_sizer.py.",
        learn_more="https://www.sec.gov/answers/stopord.htm",
        learn_more_label="Stop Order (U.S. SEC)",
    ),
    dict(
        key="transaction_cost_bps", section="Transaction costs", kind="number",
        label="Spread + slippage (bps)", min_value=0.0, max_value=200.0, step=1.0,
        help=(
            "One blended estimate of spread and slippage, applied against you on every "
            "PaperBroker fill (buys fill higher, sells fill lower than the reference price). "
            "Only affects simulated fills — AlpacaBroker's fills are real and already include "
            "whatever the market actually charged. 10 bps is a conservative round number for a "
            "mostly liquid watchlist; raise it for less liquid symbols."
        ),
        learn_more="https://www.investopedia.com/terms/b/bid-askspread.asp",
        learn_more_label="Bid-Ask Spread (Investopedia)",
    ),
    dict(
        key="commission_per_trade", section="Transaction costs", kind="number",
        label="Commission per trade ($)", min_value=0.0, max_value=50.0, step=0.5,
        help=(
            "Flat fee added on top of every PaperBroker fill, separate from the fill price "
            "itself. Defaults to 0 to match Alpaca's real commission-free equities and crypto — "
            "raise it if modeling a broker that charges one."
        ),
    ),
]

# Streamlit reruns this whole script on every widget interaction, so
# calling load_settings() here (rather than importing the module-level
# SETTINGS singleton scripts use) means a save from the Settings tab is
# reflected everywhere else in this same dashboard on the very next
# rerun -- no process restart needed. See config.py's load_settings docstring.
SETTINGS = load_settings()


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


def bars_frame(db: Database, symbol: str) -> pd.DataFrame:
    bars = db.get_bars(symbol)
    return pd.DataFrame(
        {
            "ts": [b.timestamp for b in bars],
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [b.volume for b in bars],
        }
    )


def fills_frame(db: Database, symbol: str) -> pd.DataFrame:
    fills = pd.DataFrame(db.all_fills(symbol))
    if not fills.empty:
        fills["ts"] = pd.to_datetime(fills["ts"])
    return fills


def add_fill_markers(fig: go.Figure, fills: pd.DataFrame, *, row: int | None = None, marker_size: int = 12) -> None:
    kwargs = {"row": row, "col": 1} if row is not None else {}
    if fills.empty:
        return
    buys = fills[fills["side"] == "BUY"]
    sells = fills[fills["side"] == "SELL"]
    fig.add_trace(
        go.Scatter(
            x=buys["ts"], y=buys["price"], mode="markers", name="BUY fill", showlegend=row is None,
            marker=dict(symbol="triangle-up", size=marker_size, color="green"),
        ),
        **kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=sells["ts"], y=sells["price"], mode="markers", name="SELL fill", showlegend=row is None,
            marker=dict(symbol="triangle-down", size=marker_size, color="red"),
        ),
        **kwargs,
    )


def price_and_signal_figure(bars: pd.DataFrame, fills: pd.DataFrame, decisions: pd.DataFrame) -> go.Figure:
    """One figure, not two: the price panel and the quant-signal panel share
    an x-axis. Splitting them into separate st.plotly_chart calls let each
    autorange independently -- with as few as one logged decision, the
    signal panel would zoom to a microsecond-wide window around that one
    point instead of spanning the same date range as the price chart above
    it. shared_xaxes ties them together so that can't happen.
    """
    has_signal = not decisions.empty
    rows = 2 if has_signal else 1
    row_heights = [0.72, 0.28] if has_signal else [1.0]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        row_heights=row_heights,
        subplot_titles=("", "Quant model's P(next bar closes higher)") if has_signal else None,
    )
    fig.add_trace(
        go.Scatter(x=bars["ts"], y=bars["close"], name="Close", line=dict(color="#4B8BBE")), row=1, col=1
    )
    add_fill_markers(fig, fills, row=1)

    if has_signal:
        fig.add_trace(
            go.Scatter(
                x=decisions["ts"], y=decisions["prob_up"], name="Model prob_up",
                mode="lines+markers", line=dict(color="#9467bd"),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)
        fig.update_yaxes(range=[0, 1], row=2, col=1)

    fig.update_layout(
        height=560 if has_signal else 420, margin=dict(t=40, b=20), legend=dict(orientation="h")
    )
    return fig


def small_symbol_figure(bars: pd.DataFrame, fills: pd.DataFrame, symbol: str) -> go.Figure:
    """Compact, legend-free version of the price chart for the all-symbols
    overview grid -- same data, sized to sit four-up on one screen.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bars["ts"], y=bars["close"], name="Close", line=dict(color="#4B8BBE")))
    add_fill_markers(fig, fills, marker_size=9)
    fig.update_layout(height=260, margin=dict(t=36, b=10, l=10, r=10), title=symbol, showlegend=False)
    return fig


def latest_decisions_table(db: Database, symbols: list[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        recent = db.recent_decisions(s, limit=1)
        if recent:
            r = recent[0]
            rows.append(
                {
                    "symbol": s, "as_of": r["ts"], "action": r["action"],
                    "confidence": round(r["confidence"], 2), "prob_up": round(r["prob_up"], 3),
                }
            )
        else:
            rows.append({"symbol": s, "as_of": None, "action": "no decision yet", "confidence": None, "prob_up": None})
    return pd.DataFrame(rows)


def render_settings_field(spec: dict, current: object) -> object:
    """Render one Settings-tab field's widget + help text + learn-more
    link, and return its parsed value. Call this INSIDE the bordered
    container/column you want it drawn into -- Streamlit places widgets
    wherever the currently-open `with` block is, so the field's input,
    its explanation, and its link all end up inside the same card
    instead of split across separate columns.
    """
    key = spec["key"]
    if spec["kind"] == "tags":
        raw = st.text_input(spec["label"], value=", ".join(current), key=f"field_{key}")
        value: object = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    elif spec["kind"] == "text":
        value = st.text_input(spec["label"], value=str(current), key=f"field_{key}")
    elif spec["kind"] == "number":
        is_decimal = key in DECIMAL_FIELDS
        widget_value = float(current) if is_decimal else current
        result = st.number_input(
            spec["label"], min_value=spec["min_value"], max_value=spec["max_value"],
            value=widget_value, step=spec["step"], key=f"field_{key}",
        )
        value = Decimal(str(result)) if is_decimal else int(result)
    elif spec["kind"] == "slider":
        is_decimal = key in DECIMAL_FIELDS
        result = st.slider(
            spec["label"], min_value=spec["min_value"], max_value=spec["max_value"],
            value=float(current), step=spec["step"], key=f"field_{key}",
        )
        value = Decimal(str(result)) if is_decimal else result
    elif spec["kind"] == "select":
        options = spec["options"]
        value = st.selectbox(
            spec["label"], options=options, index=options.index(current), key=f"field_{key}",
        )
    elif spec["kind"] == "toggle":
        value = st.toggle(spec["label"], value=bool(current), key=f"field_{key}")
    else:
        raise ValueError(f"unknown settings field kind: {spec['kind']!r}")

    st.caption(spec["help"])
    if spec.get("learn_more"):
        st.markdown(f"[{spec['learn_more_label']} ↗]({spec['learn_more']})")
    return value


# A scheduled script (fetch_data.py, run_paper_trader.py) that silently
# stops running -- a closed laptop, a crashed process -- is a different,
# more basic failure than a bad decision, and nothing else here would
# ever surface it. This is deliberately just "check the dashboard,"
# not push alerting: this app has no email/SMS/Slack integration to
# send through, and building one is a separate feature, not a cheap one.
_STALE_AFTER = timedelta(days=3)  # generous: covers a normal Fri-run/Mon-expected weekend gap


def render_heartbeat_status(db: Database, source: str, label: str) -> None:
    hb = db.latest_heartbeat(source)
    if hb is None:
        st.info(f"**{label}**: never run yet.")
        return

    ts = datetime.fromisoformat(hb["ts"])
    age = datetime.now(UTC) - ts
    when = f"{age.days}d {age.seconds // 3600}h ago" if age.days else f"{age.seconds // 3600}h ago"

    if hb["status"] == "error":
        st.error(f"**{label}**: last run {when} FAILED — {hb['detail']}")
    elif age > _STALE_AFTER:
        st.warning(f"**{label}**: last successful run was {when} — overdue, check the scheduled job is running.")
    else:
        st.success(f"**{label}**: last run {when}, ok.")


st.title("🌉 Ironbridge Trader")
st.caption(
    "Paper trading only — no real broker, no real money. "
    "A quant model predicts, a decision service decides, a deterministic risk layer has veto power."
)

# -- data source picker --------------------------------------------------
backtest_path = SETTINGS.data_dir / BACKTEST_DB_NAME
mode_options = ["Live paper trading"]
if backtest_path.exists():
    mode_options.append("Backtest")
mode = st.sidebar.radio("Data source", mode_options)
db = get_db(str(SETTINGS.db_path if mode == "Live paper trading" else backtest_path))

if mode == "Backtest":
    st.sidebar.warning(
        "Backtest uses the deterministic threshold service (not Claude) and "
        "the *current* model trained on the full history — results are optimistic "
        "and are for exploring decision flow, not a performance claim. "
        "See orchestration/run_backtest.py and the Model tab's held-out accuracy for the honest number."
    )

symbols = db.symbols_with_bars()
if not symbols:
    st.warning("No data yet. Run `python scripts/fetch_data.py` first.")
    st.stop()
symbol = st.sidebar.selectbox("Symbol (for the per-symbol tabs below)", symbols)

if SETTINGS.anthropic_api_key:
    st.sidebar.success("ANTHROPIC_API_KEY set — live cycles use the Claude tool-using agent.")
else:
    st.sidebar.info("No ANTHROPIC_API_KEY — live cycles use the deterministic threshold fallback.")

if mode == "Live paper trading":
    if not SETTINGS.trading_enabled:
        st.warning("⏸️ **Trading is paused** (kill switch is off) — flip it back on in the ⚙️ Settings tab.")
    hb_cols = st.columns(2)
    with hb_cols[0]:
        render_heartbeat_status(db, "fetch_data", "Data fetch")
    with hb_cols[1]:
        render_heartbeat_status(db, "run_paper_trader", "Decision cycle")

tab_overview, tab_flow, tab_decisions, tab_model, tab_portfolio, tab_settings = st.tabs(
    ["Overview (all symbols)", "Price & Signal", "Agent Decisions", "Model", "Portfolio", "⚙️ Settings"]
)

# -- Tab 0: every symbol at a glance ---------------------------------------
with tab_overview:
    st.subheader("Latest decision, every symbol")
    st.dataframe(latest_decisions_table(db, symbols), use_container_width=True, hide_index=True)

    st.subheader("Price & fills, every symbol")
    cols_per_row = 2
    for i in range(0, len(symbols), cols_per_row):
        row_symbols = symbols[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, s in zip(cols, row_symbols):
            with col:
                st.plotly_chart(small_symbol_figure(bars_frame(db, s), fills_frame(db, s), s), use_container_width=True)

# -- Tab 1: price chart with buy/sell markers + quant signal, one symbol --
with tab_flow:
    bars = bars_frame(db, symbol)
    decisions = pd.DataFrame(db.all_decisions(symbol))
    fills = fills_frame(db, symbol)
    if not decisions.empty:
        decisions["ts"] = pd.to_datetime(decisions["ts"])

    st.plotly_chart(price_and_signal_figure(bars, fills, decisions), use_container_width=True)
    if decisions.empty:
        st.info("No decisions logged yet for this symbol.")

# -- Tab 2: the decision log, in the agent's own words --------------------
with tab_decisions:
    st.subheader(f"Decision log — {symbol}")
    if decisions.empty:
        st.info("Run `python scripts/run_paper_trader.py` (or the backtest) to generate decisions.")
    else:
        show = decisions.sort_values("ts", ascending=False)[
            ["ts", "action", "confidence", "prob_up", "agent_kind", "rationale"]
        ]
        st.dataframe(show, use_container_width=True, hide_index=True)

# -- Tab 3: model version history -----------------------------------------
with tab_model:
    st.subheader("Model versions")
    versions = pd.DataFrame(db.model_versions())
    if versions.empty:
        st.info("No trained model yet. Run `python scripts/train_model.py`.")
    else:
        versions["trained_at"] = pd.to_datetime(versions["trained_at"])
        st.caption(
            "eval_accuracy / eval_auc are computed on a chronologically-held-out slice "
            "(walk-forward split, no lookahead) — this is the honest accuracy number."
        )
        metric_fig = go.Figure()
        metric_fig.add_trace(go.Scatter(x=versions["trained_at"], y=versions["eval_accuracy"], name="Accuracy"))
        metric_fig.add_trace(go.Scatter(x=versions["trained_at"], y=versions["eval_auc"], name="AUC"))
        metric_fig.update_layout(height=300, margin=dict(t=20, b=20), yaxis_range=[0, 1])
        st.plotly_chart(metric_fig, use_container_width=True)
        st.dataframe(versions.sort_values("trained_at", ascending=False), use_container_width=True, hide_index=True)

# -- Tab 4: portfolio / equity ---------------------------------------------
with tab_portfolio:
    st.subheader("Account performance vs. buy & hold")
    st.caption(
        "The account-wide equity curve — cash plus mark-to-market value across "
        "every symbol — against a zero-decisions, fully-invested-on-day-one buy-and-hold "
        "benchmark of the same watchlist. This is the bar an active strategy has to "
        "clear, after every decision and every trade, to be worth the complexity."
    )
    strategy_equity = performance.compute_equity_curve(db, symbols, SETTINGS.account_equity)
    benchmark_equity = performance.compute_buy_and_hold_curve(db, symbols, SETTINGS.account_equity)

    if strategy_equity.empty:
        st.info("No fills yet — run the paper trader or backtest to generate an equity curve.")
    else:
        perf_fig = go.Figure()
        perf_fig.add_trace(go.Scatter(x=strategy_equity.index, y=strategy_equity.values, name="Strategy"))
        if not benchmark_equity.empty:
            perf_fig.add_trace(
                go.Scatter(x=benchmark_equity.index, y=benchmark_equity.values, name="Buy & hold")
            )
        perf_fig.update_layout(
            height=320, margin=dict(t=20, b=20), legend=dict(orientation="h"), title="Account equity ($)"
        )
        st.plotly_chart(perf_fig, use_container_width=True)

        drawdown = performance.compute_drawdown(strategy_equity)
        dd_fig = go.Figure(
            go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", fill="tozeroy", line=dict(color="#d62728"))
        )
        dd_fig.update_layout(
            height=200, margin=dict(t=20, b=20), title="Strategy drawdown", yaxis_tickformat=".0%"
        )
        st.plotly_chart(dd_fig, use_container_width=True)

        strat_return = strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1
        bench_return = (
            benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1 if not benchmark_equity.empty else None
        )
        strat_mdd = performance.max_drawdown(strategy_equity)
        strat_sharpe = performance.sharpe_ratio(strategy_equity)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Strategy return", f"{strat_return:.1%}",
            delta=f"{(strat_return - bench_return):+.1%} vs. buy & hold" if bench_return is not None else None,
        )
        m2.metric("Buy & hold return", f"{bench_return:.1%}" if bench_return is not None else "—")
        m3.metric("Max drawdown", f"{strat_mdd:.1%}" if strat_mdd is not None else "—")
        m4.metric("Sharpe ratio", f"{strat_sharpe:.2f}" if strat_sharpe is not None else "—")
        st.caption(
            "Sharpe uses raw returns with no risk-free-rate adjustment, and 252 trading "
            "days/year for annualizing — an approximation for a watchlist that also "
            "holds BTC-USD, which trades 365 days/year. See analytics/performance.py."
        )

    st.divider()
    st.subheader(f"Position over time — {symbol}")
    equity = pd.DataFrame(db.equity_curve(symbol))
    if equity.empty:
        st.info("No fills yet for this symbol.")
    else:
        equity["ts"] = pd.to_datetime(equity["ts"])
        c1, c2 = st.columns(2)
        with c1:
            qty_fig = go.Figure(go.Scatter(x=equity["ts"], y=equity["position_qty"], name="Position qty"))
            qty_fig.update_layout(height=280, margin=dict(t=20, b=20), title="Position size")
            st.plotly_chart(qty_fig, use_container_width=True)
        with c2:
            val_fig = go.Figure(go.Scatter(x=equity["ts"], y=equity["position_value"], name="Position value"))
            val_fig.update_layout(height=280, margin=dict(t=20, b=20), title="Position value ($)")
            st.plotly_chart(val_fig, use_container_width=True)

    st.subheader("All fills")
    all_fills = pd.DataFrame(db.all_fills(symbol))
    if not all_fills.empty:
        st.dataframe(all_fills.sort_values("ts", ascending=False), use_container_width=True, hide_index=True)

# -- Tab 5: settings -- policy knobs, never secrets ------------------------
with tab_settings:
    st.subheader("Configuration")
    st.caption(
        "Operations panel — safe to leave open on a shared screen: this page can only "
        "read and write non-secret policy values (data/settings_overrides.json). It never "
        "shows or edits ANTHROPIC_API_KEY, which stays in .env."
    )
    st.caption(
        "Read this before turning anything: this app's own measured accuracy is close to "
        "a coin flip (see README's Trading theory section). These are risk-policy and "
        "operational knobs, not a way to make the model smarter — each one links to the "
        "underlying concept if you want to understand it before changing it."
    )

    api_key_set = bool(SETTINGS.anthropic_api_key)
    badge_cols = st.columns(4)
    _badges = [
        ("Anthropic API key", "configured" if api_key_set else "not set", api_key_set),
        ("Model", SETTINGS.anthropic_model, None),
        ("Watchlist", f"{len(SETTINGS.symbols)} symbols", None),
        ("Overrides file", "data/settings_overrides.json", None),
    ]
    for col, (label, value, ok) in zip(badge_cols, _badges):
        color = "#4ade80" if ok else ("#f87171" if ok is False else "#60a5fa")
        col.markdown(
            f"<div style='border:1px solid {color}55;border-radius:8px;padding:8px 12px;'>"
            f"<div style='color:#888;font-size:0.75em;text-transform:uppercase;'>{label}</div>"
            f"<div style='color:{color};font-weight:600;'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    sections: dict[str, list[dict]] = {}
    for spec in SETTINGS_FIELDS:
        sections.setdefault(spec["section"], []).append(spec)

    # Each field is its own bordered card: widget, help text, and its
    # learn-more link all together in one box, two cards per row. Splitting
    # inputs into one column and every description into a separate column
    # (the previous layout) reads as a disconnected wall of text once more
    # than two or three fields are stacked -- nothing here should require
    # counting rows to match a sentence back to the slider it describes.
    form_values: dict = {}
    with st.form("settings_form"):
        for section_name, specs in sections.items():
            st.markdown(f"#### {section_name}")
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                with cols[i % 2], st.container(border=True):
                    form_values[spec["key"]] = render_settings_field(spec, getattr(SETTINGS, spec["key"]))
            st.write("")

        st.divider()
        submitted = st.form_submit_button("💾 Save settings", type="primary")

    if submitted:
        save_overrides(form_values)
        st.success(
            "Saved. This dashboard picks it up on your next click here; scripts "
            "(fetch_data.py, train_model.py, run_paper_trader.py, backtest.py) pick it up "
            "the next time you run them, since each is a fresh process."
        )
        st.rerun()
