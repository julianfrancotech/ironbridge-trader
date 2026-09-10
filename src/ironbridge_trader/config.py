"""Central, explicit configuration. One dataclass, populated from env vars
with sane defaults -- no config framework, no YAML, no layered overrides
(a centralized secrets manager solves a many-services, many-environments
problem this single-user local app doesn't have).
This is the one file you edit to change the watchlist or risk limits --
or, for the fields listed in TUNABLE_FIELDS, the dashboard's Settings
tab, which writes to data/settings_overrides.json rather than .env.
That file is kept separate from .env on purpose: .env holds the one
secret (ANTHROPIC_API_KEY), and a settings UI that's safe to leave open
during screen sharing should never be the thing that can read or write
a secret -- see dashboard/app.py's Settings tab.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = PROJECT_ROOT / "data" / "settings_overrides.json"

# Fields the Settings UI is allowed to persist. Deliberately excludes:
#   - anthropic_api_key: a secret, edited via .env only, never through a
#     UI meant to be safe to screen-share.
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
    "max_position_size",
    "margin_rate",
    "risk_fraction",
    "stop_loss_fraction",
    "max_concurrent_symbols",
)

DECIMAL_FIELDS = {"account_equity", "margin_rate", "risk_fraction", "stop_loss_fraction"}


@dataclass(frozen=True, slots=True)
class Settings:
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
    max_position_size: int = 100
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

    # How many symbols run_cycle / fetch_market_data process in parallel
    # (see engine/trading_engine.py, orchestration/fetch_market_data.py).
    # Bounded so a large watchlist doesn't thundering-herd past the
    # Anthropic API's or a broker's own rate limits.
    max_concurrent_symbols: int = 5

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "ironbridge_trader.db")
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")

    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

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
