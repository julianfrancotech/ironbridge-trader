"""Central, explicit configuration. One dataclass, populated from env vars
with sane defaults -- no config framework, no YAML, no layered overrides
(a centralized secrets manager solves a many-services, many-environments
problem this single-user local app doesn't have).
This is the one file you edit to change the watchlist or risk limits --
or, for the fields listed in TUNABLE_FIELDS, the dashboard's Settings
tab, which writes to data/settings_overrides.json rather than .env.
That file is kept separate from .env on purpose: .env holds the app's
secrets (ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY), and a
settings UI that's safe to leave open during screen sharing should
never be the thing that can read or write a secret -- see
dashboard/app.py's Settings tab.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = PROJECT_ROOT / "data" / "settings_overrides.json"

# Fields the Settings UI is allowed to persist. Deliberately excludes:
#   - anthropic_api_key / alpaca_api_key / alpaca_secret_key: secrets,
#     edited via .env only, never through a UI meant to be safe to
#     screen-share.
#   - bar_interval: an architectural assumption baked into
#     MIN_BARS_REQUIRED and the walk-forward split, not a casual dial --
#     see docs/platform-boundaries.md.
#   - data_dir / db_path / model_dir: filesystem layout, not trading policy.
TUNABLE_FIELDS = (
    "symbols",
    "history_years",
    "feature_lookback",
    "train_test_split_ratio",
    "decision_confidence_floor",
    "agent_max_tool_iterations",
    "anthropic_model",
    "account_equity",
    "max_position_fraction",
    "margin_rate",
    "risk_fraction",
    "stop_loss_fraction",
    "transaction_cost_bps",
    "commission_per_trade",
    "max_concurrent_symbols",
    "data_provider",
    "execution_provider",
    "trading_enabled",
)

DECIMAL_FIELDS = {
    "account_equity", "max_position_fraction", "margin_rate", "risk_fraction", "stop_loss_fraction",
    "transaction_cost_bps", "commission_per_trade",
}


@dataclass(frozen=True, slots=True)
class Settings:
    # Manual kill switch, checked once at the top of
    # orchestration/run_decision_cycle.py::run() -- flip to False to
    # stop live trading immediately, independent of and faster than any
    # code change. Deliberately does NOT gate scripts/backtest.py: a
    # paused live strategy shouldn't also block research/backtesting.
    trading_enabled: bool = True

    # Watchlist: a small, liquid, mixed set by default (equities + one
    # index ETF + crypto, which trades on weekends too). Edit freely.
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "SPY", "BTC-USD")

    # Daily bars, not intraday ticks -- see docs/platform-boundaries.md.
    bar_interval: str = "1d"
    history_years: int = 5

    # ML: labels are "next bar's close > this bar's close".
    feature_lookback: int = 60  # bars of history the feature engineer needs
    train_test_split_ratio: float = 0.8  # walk-forward: first 80% train, last 20% eval

    # Decision service: below this confidence, or on a tie, the decision is HOLD.
    decision_confidence_floor: float = 0.55
    agent_max_tool_iterations: int = 6
    anthropic_model: str = "claude-sonnet-5"

    # Risk (deterministic gate, applied to every proposed order regardless
    # of which DecisionMaker produced it):
    account_equity: Decimal = Decimal(100_000)
    # Hard cap on one symbol's position, as a fraction of account_equity
    # (notional value) -- not a fixed unit count. See risk/manager.py's
    # docstring: a fixed count doesn't scale with price, and a real
    # backtest confirmed that's not a theoretical concern (see
    # scripts/risk_veto_report.py). 0.25 is a backstop, not the primary
    # sizing mechanism -- risk_fraction below is what actually drives
    # typical position sizes; this just bounds the worst case.
    max_position_fraction: Decimal = Decimal("0.25")
    margin_rate: Decimal = Decimal("0.25")

    # Position sizing (risk/position_sizer.py): how many units to buy/sell
    # is DERIVED from account risk tolerance, not a fixed quantity that's
    # the same regardless of price or account size. risk_fraction is the
    # fraction of account_equity you're willing to lose on one trade;
    # stop_loss_fraction is the assumed adverse-move distance used only
    # for that sizing math -- this app does not place an actual stop
    # order, see risk/position_sizer.py's docstring.
    risk_fraction: Decimal = Decimal("0.01")
    stop_loss_fraction: Decimal = Decimal("0.05")

    # Transaction costs (adapters/paper_broker.py, analytics/performance.py):
    # PaperBroker fills are synthetic, so without these every backtest and
    # paper-trading number implicitly assumes zero spread, zero slippage,
    # zero commission -- an assumption real trading never gets. Bar data
    # has no bid/ask to model spread and slippage as separate line items,
    # so transaction_cost_bps blends both into one number applied against
    # the trader (BUY fills higher, SELL fills lower than the reference
    # price); 10 bps is a conservative round-number estimate for a mostly
    # liquid-large-cap watchlist that also holds BTC-USD, which typically
    # trades wider. commission_per_trade defaults to 0 to match Alpaca's
    # real commission-free equities/crypto, kept as a knob for a future
    # broker that isn't. AlpacaBroker ignores both -- its fills are real,
    # already reflecting whatever the market and broker actually charged.
    transaction_cost_bps: Decimal = Decimal(10)
    commission_per_trade: Decimal = Decimal(0)

    # How many symbols run_cycle / fetch_market_data process in parallel
    # (see engine/trading_engine.py, orchestration/fetch_market_data.py).
    # Bounded so a large watchlist doesn't thundering-herd past the
    # Anthropic API's or a broker's own rate limits.
    max_concurrent_symbols: int = 5

    # Market data / execution providers. Both default to the free,
    # no-credentials-needed path so nothing breaks for anyone without
    # Alpaca keys. Explicit opt-in fields rather than auto-switching the
    # moment ALPACA_API_KEY is set -- unlike the decision-maker choice
    # (build_decision_maker), switching execution means real orders start
    # hitting a real broker's (paper) account, which deserves a deliberate
    # choice, not an implicit one. See adapters/alpaca_market_data.py,
    # adapters/alpaca_broker.py.
    data_provider: Literal["yfinance", "alpaca"] = "yfinance"
    execution_provider: Literal["paper_broker", "alpaca_paper"] = "paper_broker"

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "ironbridge_trader.db")
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")

    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    # Alpaca paper-trading credentials. Free to create at alpaca.markets;
    # never required -- both providers above default to the path that
    # doesn't need them.
    alpaca_api_key: str | None = field(default_factory=lambda: os.getenv("ALPACA_API_KEY"))
    alpaca_secret_key: str | None = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY"))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


def _read_overrides(path: Path = OVERRIDES_PATH) -> dict:
    """Read data/settings_overrides.json, keeping only recognized
    TUNABLE_FIELDS and converting each back to the type Settings expects
    (JSON has no Decimal or tuple). Silently ignores a missing or
    corrupt file -- a bad override should degrade to defaults, not
    crash every script and the dashboard.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    overrides = {}
    for key, value in raw.items():
        if key not in TUNABLE_FIELDS:
            continue
        if key in DECIMAL_FIELDS:
            overrides[key] = Decimal(str(value))
        elif key == "symbols":
            overrides[key] = tuple(value)
        else:
            overrides[key] = value
    return overrides


def save_overrides(values: dict, path: Path = OVERRIDES_PATH) -> None:
    """Persist a subset of TUNABLE_FIELDS so the next process to call
    load_settings() (any script, or the dashboard's next rerun) picks
    them up. Decimal values are written via str(), not float(), so they
    round-trip exactly instead of picking up binary-float noise.
    """
    serializable = {}
    for key, value in values.items():
        if key not in TUNABLE_FIELDS:
            raise ValueError(f"{key} is not a user-tunable setting")
        if isinstance(value, Decimal):
            serializable[key] = str(value)
        elif isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_settings() -> Settings:
    """The one place Settings gets constructed. Scripts call this once
    per process (each is a fresh process anyway, so a fresh read is
    free). The dashboard calls it once per Streamlit rerun -- which
    Streamlit already does on every widget interaction -- so a save
    from the Settings tab is reflected elsewhere in the same running
    dashboard without needing a restart.
    """
    return Settings(**_read_overrides())


SETTINGS = load_settings()
