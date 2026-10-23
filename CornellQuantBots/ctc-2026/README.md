# Hurricane Season bot v2

This is the CTC 2026 Systematic Equities bot. It is **dry-run by default**: it sends live orders only when you pass `--live` and also set `CTC_ALLOW_LIVE=1`.

## Install (macOS, Python 3.10+)

```bash
cd ctc_hurricane_v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the research archive next to this folder (../hurricane-season-research) or pass its path:
python library.py ../hurricane-season-research data/lib_all.npz 1-600      # analog library for live (about 1 min)
python library.py ../hurricane-season-research data/lib_train.npz 201-600  # for tests and replay
mkdir -p ~/.ctc && nano ~/.ctc/keys.txt && chmod 600 ~/.ctc/keys.txt       # one bot key per line; never commit
HURRICANE_RESEARCH=../hurricane-season-research python -m pytest -q tests   # 18 tests must pass
```

## Files

| File | Role |
|---|---|
| `simmodel.py` | Recovered simulator: wind, closures, damage exposure, settlement formulas, insurance layers. Parameters are in `data/params.json`, fitted on seasons 201–600 only. |
| `library.py`, `forecast.py` | Analog storm library; pattern posterior from all advisories (official 70%, wire 40% accurate, independent); analog path futures. |
| `valuation.py` | Joint Monte Carlo scenarios for all 12 settlements, using common random numbers. |
| `engine.py` | Feed-driven state. Validates every message, de-duplicates, handles snapshots, and has a `complete()` gate: no trading after a late start or a data gap. |
| `strategy.py` | Core policy (the one validated in backtests) plus the final-report snipe, the markout guard, and announcement/override hooks. |
| `router.py` | The only code that sends order actions. Tracks the ids it owns, checks capacity against team positions and all resting team orders, recovers timed-out POSTs, never touches teammates' orders, never calls `DELETE /orders/all`. |
| `ratelimit.py` | Rate limiter shared through a file lock across all our processes. `--team-share` sets our slice of the team's 180 actions per minute. |
| `marketdata.py` | WebSocket client: reconnect, seq de-duplication, stale and empty quote detection, REST book fallback. |
| `announcements.py` | Rule-based announcement parser. Auto-prices only exact numeric statements, otherwise pauses and alerts. **Uncalibrated.** |
| `etf.py` | Arbitrage on executable full-depth books for ARCH and REBLD (off by default). |
| `control.py` | Human control: pause, kill, per-symbol pause, size, fair overrides, parameter and adjustment changes, ETF on/off. |
| `bot.py` | Orchestrator. |
| `replay.py` | Replays research seasons through the live code path. |
| `research/` | The backtester and experiment scripts behind REPORT.md (paths inside are those of the research machine). |

See RUNBOOK.md for competition day and REPORT.md for the evidence.
