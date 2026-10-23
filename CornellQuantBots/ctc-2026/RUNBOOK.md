# Competition-day runbook

## Before the event (once)

1. Run the install steps in README.md, then `python -m pytest -q tests`. All 18 tests must pass.
2. Competition day: **this one process is the only trader on the team account and drives all three keys** (`--sole-trader`).
   - It uses up to 170 of the team's 180 actions per minute (57 per key).
   - It opens one market-data connection per key; duplicate messages are dropped by seq, so a message is lost only if all three feeds miss it.
   - Any resting order it finds that it didn't place is treated as its own (left over from a crash) and is cancelled.
   - The kill switch uses `DELETE /orders/all`.
   - **No human or other bot should place orders on the account while it runs.** If someone must trade by hand, first run `python control.py pause-sym SYMBOL 600` for the symbols they'll touch.
3. Put all three keys in `~/.ctc/keys.txt`, one per line.

## Launch (start BEFORE Round 1 opens)

```bash
cd ctc_hurricane_v2 && source .venv/bin/activate
export CTC_RUN_DIR=$PWD/run_live
# 1) dry run for a minute: check the status line, 'complete': True once a storm starts, and no *_err lines
python bot.py --mode live --sole-trader
# 2) live
CTC_ALLOW_LIVE=1 python bot.py --mode live --live --sole-trader 2>&1 | tee -a run_live/console.txt
```

Use `--mode live` on the day: the analog library uses all 600 research seasons. `--mode testnet` instead excludes whichever research season the testnet is replaying.

## While running

- **Status line every 10 s:** fictional time, phase info, rate usage, and position@mid/fair for each symbol.
- **Log:** `run_live/bot.jsonl`, one JSON object per line. Useful filters:
  - `grep ALERT run_live/bot.jsonl`
  - `grep '"kind": "fill"' run_live/bot.jsonl`
  - `grep _err run_live/bot.jsonl`
- **`'complete': False`** means the bot started late or lost data. It will not trade until the next round's storm has been seen from age 0.
- **Announcements** print a loud banner.
  - "AUTO-PRICED": check the parsed number. Undo with `python control.py reset-adjustments`.
  - "NOT PRICED": the symbols are paused for 120 s. If you can price it, set the value yourself:
    - `python control.py fair HAVEN 41.5 60` sets HAVEN's fair value to 41.5 for 60 s.
    - `python control.py param HAVEN_ATT 500` changes an insurance-layer parameter.
    - `python control.py adjust PORT add -3` adds −3 per share to PORT's settlement value.
    - `python control.py unpause-sym HAVEN` lifts the pause.

## Stopping and disabling trading

| Need | Command (in another terminal, same `CTC_RUN_DIR`) |
|---|---|
| Stop new orders, keep positions | `python control.py pause`, then later `python control.py resume` |
| Pull one symbol | `python control.py pause-sym GRID 600` |
| Trade smaller | `python control.py size 0.5` |
| Emergency stop (cancels ALL resting orders on the account) | `python control.py kill`, then Ctrl-C the bot. Before restarting it later: `python control.py unkill` |
| Bot hung | Ctrl-C. Resting orders stay on the book. Cancel them in the UI, or restart with `--live --sole-trader` and immediately run `python control.py kill`. A restart in sole-trader mode also adopts and cancels them. |

## Known limits

- **The edge depends on how fast competitors arbitrage the house.** In the backtest the core strategy averages about +$7k per season with moderate competition and about $0 when half of the edge is already in the price (REPORT.md, section 4). The markout guard shrinks sizes automatically when our fills keep losing.
- **Unverified API behavior; test once on the testnet before relying on it:**
  - ETF create/redeem.
  - Whether the 6th-decimal price signature survives on the exchange.
- The ETF module is off unless the bot is started with `--enable-etf` and you also run `python control.py etf on`.
