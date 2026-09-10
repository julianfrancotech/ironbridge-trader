# Platform boundaries

Modeled on the kind of platform-boundaries doc a larger production
trading service keeps: an explicit, written statement of what this app
does and doesn't know about, so scope stays a decision instead of an
accident.

## What this app knows about

- **A configured watchlist** of symbols (`config.py`), each with its own
  stored bar history, model predictions, decisions, and paper positions.
- **One account**: a single simulated equity balance and one set of risk
  limits, shared across the whole watchlist. There is no multi-tenant
  account model, no organizations, no per-user permissions.
- **One decision per symbol per bar.** The app does not attempt to reason
  about correlation or portfolio-level risk across symbols — each
  symbol's risk check only looks at that symbol's own position and the
  shared account equity.

## What this app deliberately does not know about

- **Real brokerage accounts or real money.** `adapters/paper_broker.py`
  fills every order at its reference price with no real execution risk.
  Wiring in a real broker is a one-class change (satisfy
  `protocols.ExecutionClient`) that is intentionally left undone.
- **Other users.** There is no auth, no identity service, no per-user
  data isolation — this is a single-operator tool, not a multi-tenant
  platform.
- **Other upstream business services** (accounts, authz, identity,
  product catalog). Nothing here needs to be aware of an organization
  or a legal entity; the "account" is just a number in `config.py`.

## Why this app has no HTTP/API layer

A larger production trading service typically has a FastAPI (or similar)
layer because *other services* (admin tooling, integration tests, other
internal consumers) call it over the network, and because decoupling the
API from a persistent exchange connection is a real scaling need there.
Neither is true here: nothing else consumes
this app, and there is no persistent connection to decouple anything
from — market data is a periodic HTTP pull (`adapters/market_data.py`),
not a held-open session. The Streamlit dashboard imports the package
directly. If that ever changes (see below), an API layer is where to
start.

## What would need to change if this ever went from daily-batch to live/intraday

This is the direct answer to "what do we need for this to scale,"
without building any of it prematurely:

1. **A persistent adapter connection**, the way a real exchange adapter
   holds a websocket open — at that point an explicit connection state
   machine (`DISCONNECTED -> CONNECTED -> ...`) stops being ceremony and
   starts being necessary, because a dropped connection is now a real
   failure mode mid-session instead of "the next scheduled pull will
   just retry."
2. **Decoupling the always-on price/decision process from anything
   request/response** (a dashboard, an eventual API) — this is exactly
   the kind of problem a multi-runtime-mode split (a dedicated pricing
   process, a dedicated order-processing process, a separate API
   process) exists to solve.
3. **A background job runner** if decisions need to happen faster than a
   cron job can be scheduled, or if event-driven triggers (a price
   crossing a threshold) replace "once a day."
4. **Concurrent-write-safe storage.** SQLite's single-writer model is
   fine for one cron job at a time; a live process making frequent
   writes while the dashboard reads would want Postgres.

None of this is needed for the batch, single-operator, daily-bar app
that exists today — which is exactly why none of it is built.
