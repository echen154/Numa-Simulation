# Systematic Equities: audit, research and v2 results

All numbers below come from runs on 2026-10-22. Data partitions are chronological:

| Partition | Seasons | Used for |
|---|---|---|
| Train | 201–600 | Every fitted parameter and the analog library |
| Validation | 101–200 | Choosing policies and hyperparameters (V1 = 101–150 for tuning, V2 = 151–200 for checking) |
| Test | 1–100 | Run once, at the end |

Testnet observations are read-only: recorded books and quotes, plus team positions. **No orders were sent during this work.**

## 1. Audit of the original bot (most severe first)

| # | Finding | Where | Consequence |
|---|---|---|---|
| 1 | **Unsafe on a shared account.** `GET /positions` and `GET /orders` are **team-wide**: all three keys return identical results, and positions changed while only a teammate traded. v1 sized against team positions as if they were its own. `sync_loop` kept one order per symbol, read everything with key 0, had no team-wide rate limit, no handling of timed-out POSTs, and a token hack in `replace()`. | `bot.py` `sync_loop`, `trade_once`, `replace` | Would trade against or unwind teammates' positions, could cancel their orders, and could exceed the 180-actions-per-minute team limit. |
| 2 | **Correlated over-concentration.** Each symbol was sized independently on the same storm forecast, so all 12 positions formed one bet. In testnet season 202 it held near-full limits in 11 symbols during round 1. Held to settlement, those positions would have lost about **$10k** (−$3.3k marked when stopped). | `bot.py` `target` | Fat left tail: worst test season −$16k in the base execution scenario. |
| 3 | **Optimistic evaluation.** <ul><li>Half-spreads came from one snapshot, fills of 150 shares at house ± half-spread, no competitors, no latency.</li><li>All stochastic parameters were fitted on all 600 seasons.</li><li>Hyperparameters were tuned on part of the "held-out" set.</li><li>The +$29.7k testnet season was in-sample.</li></ul> | `simpnl.py`, `tune.py` | The reported $7.4–7.8k per season was not a held-out number; see section 4 for the corrected comparison. |
| 4 | **Malformed feed message crashed the handler** (non-numeric `age`). | `engine.py` `add_obs` | A loop-level exception rather than a skipped message. Fixed and tested in v2. |
| 5 | **Late start without history.** The v1 dry run showed a fair HAVEN value of 60.7 against a true ~30 when it joined mid-season. v1 already had a completeness gate; v2 keeps it, with stricter coverage checks. | `engine.py` | Prevented. |
| 6 | **Leakage, by item.** <ul><li>Settlement formulas, insured-loss coefficients, wind/closure threshold, damage rates and noise, advisory accuracies and pattern priors, preliminary-report blend weight: all 600 seasons.</li><li>Analog library: all seasons except the one being evaluated.</li><li>Beta schedule: tuned on half of the "held-out" set.</li></ul> **Refitting everything on seasons 201–600 gives practically identical parameters** (closure threshold 0.35, wind k = 1.0, rates within 5%, intensity exponent 0.84→0.90, official 0.698, wire 0.395), so leakage barely mattered. Execution assumptions did. | — | Measured, not assumed. |

## 2. Recovered simulator: verification on untouched seasons 1–100

| Component | Status | Evidence |
|---|---|---|
| Settlement of all 12 symbols, given final damage and closures | **Exact** (to the cent) | Mean absolute error 0.002–0.014 per stock (ARCH 0.031, from the closure discretisation). Max error 0.12. No bias. |
| Insured loss per storm = 9·PORT + 6·ISLE + 2.5·AQUA + 7·GRID + 1.2·FERRY (damage points, $M) | **Exact** | Residual sd 0.04 $M |
| Insurance layers: HAVEN pays $400M–$1,300M at $0.06/$M; REIN pays above $1,300M at $0.035/$M | **Exact** | Max error $0.01 |
| Wind = intensity · exp(−(d/r)²); closed while > 0.35 | Recovered (to 1 s simulator steps) | PORT residual sd 0.012 |
| Damage = remaining · (1 − exp(−rate · ε · Σ I^Q · exp(−(d/r)²))) | **Stochastic** | Company rmse 2.4–5.3 points; log-noise sd 0.25–0.30, independent across companies (correlation < 0.1) |
| Preliminary report | Noisy | Error sd ≈ 30%. Optimal blend weight 0.19–0.25 on the preliminary, the rest on the track model |
| Official outlook / wire rumour | Independent noisy signals | 70% / 40% accurate at every age. Combined, the posterior is right 87% of the time by age 120 s and 95% by 300 s |
| Official cone | Better than analog paths | Position error 13/26/47/95 vs 18/34/62/133 map units at 40/80/140/240 s horizons |

## 3. Strategy research: what survived

| Idea | Result | Kept? |
|---|---|---|
| **House lag during storms** | The house closes about 40% of the gap to our fair value within 60–120 s at ages 160–480 s. Settlement regressed on our signal gives a slope of about 0.6–1.0 at ages 200–400 s and about 0 after age 420 s. This is the core edge. | **Yes** (core) |
| Trust the model fully (beta = 1) | Loses heavily in every execution scenario: the house knows things we don't. | No |
| Cone-adjusted forecaster | Better settlement MAE early in storms, but mixed for trading: test base +$281, conservative +$541 vs v1, worse on validation. | Optional, not default |
| Scenario portfolio allocator (correlation-aware) | Much smaller tails (harsh scenario −$153 vs −$1,751), but 40% lower mean in normal conditions. | No: research code only (`research/policies.py`) |
| Short-term repricing trades (enter mid-storm, exit after the storm) | Lower mean (−$2.2k per season vs holding) and lower sd. | No |
| **Insurance-layer mispricing** | The house's HAVEN, REIN and MAT prices match the house's own damage estimates on the physical stocks (REIN off by about $0.1). The real insurance edge comes from forecasting: HAVEN is the largest and most robust contributor (+$2.0k per season base; +$400 even in the conservative scenario). | Through the core |
| **Final-report snipe** | Testnet, season 202, round 3: HAVEN quoted about $1.00 above exact settlement for about 4 s after the report, GRID $0.25–0.35 for about 6 s. ARCH corrected within 0.6 s (another team is fast). | **Yes** (urgent path) |
| **ETF create/redeem arbitrage** | The CSV-based estimate (~$6k per season) was illusory. On executable testnet books, ARCH was profitable in 27% of 252 snapshots (mean $11.6, max $56, episodes about 10 s); REBLD rarely. Realistic value: hundreds of dollars per season. | Built, **off by default**: create/redeem semantics unverified |
| Inventory skew of house quotes | Testnet book mids sat $0.07–1.6 away from the house consensus, driven by other teams' flow. Our impact model uses 0.5–1.5 × half-spread per top-of-book size. | In the simulator |
| Announcements | No examples exist. Deterministic parser auto-prices only explicit numbers mapped to known parameters; everything else pauses and alerts. | Yes, **uncalibrated** |

## 4. Before and after: same seasons, simulator, execution and accounting (test seasons 1–100)

Execution scenarios (spreads, depth and timing calibrated to the testnet recording):

| Scenario | Storm spread | Depth | Impact | Crowding | Latency | Our actions/min |
|---|---|---|---|---|---|---|
| Optimistic | 1× | 1.5× | 0 | 0 | 2 s | 120 |
| Base | 2× | 1× | 0.5 | 25% of edge already priced | 2 s | 100 |
| Conservative | 3.5× | 0.6× | 1.0 | 50% | 4 s | 80 |
| Harsh | 5× | 0.4× | 1.5 | 70% | 6 s | 60 |

Results ($ per season; 10 symbols plus 2 ETFs; positions settle at fundamental value):

| Scenario | Policy | Mean | Median | Win rate | p10 | p5 | Worst | Actions |
|---|---|---|---|---|---|---|---|---|
| Optimistic | v1 core (old bot logic, train-only model) | 16,026 | 15,299 | 0.97 | 4,973 | 1,857 | −4,134 | 1,477 |
| Optimistic | settle-only (beta = 1) | 10,462 | 10,813 | 0.74 | −8,055 | −14,051 | −29,409 | 1,566 |
| Optimistic | scenario allocator | 9,505 | 9,249 | 0.98 | 2,274 | 1,769 | −788 | 673 |
| Base | **v1 core** | **7,090** | 6,434 | 0.79 | −4,868 | −7,264 | −16,157 | 2,179 |
| Base | v1 + cone | 7,371 | 7,242 | 0.76 | −7,103 | −8,817 | −14,034 | 2,177 |
| Base | settle-only | −5,226 | −3,848 | 0.40 | −23,559 | −28,181 | −43,342 | 2,780 |
| Base | window / unwind | 4,872 | 5,395 | 0.80 | −3,710 | −6,845 | −15,032 | 1,212 |
| Base | scenario allocator | 4,140 | 4,744 | 0.76 | −4,608 | −7,243 | −11,692 | 734 |
| Conservative | v1 core | −258 | 328 | 0.51 | −9,839 | −13,393 | −27,769 | 2,233 |
| Conservative | scenario allocator | 94 | 404 | 0.58 | −4,202 | −6,703 | −16,321 | 452 |
| Harsh | v1 core | −1,751 | −1,319 | 0.45 | −8,226 | −10,936 | −27,348 | 1,432 |
| Harsh | scenario allocator | −153 | −33 | 0.47 | −2,162 | −2,540 | −9,262 | 232 |

House baseline (no trades) is $0 in every scenario. The full table is in `research/final_test_results.csv`.

**Interpretation:**
- **v2's directional engine is the v1 policy.** No forecasting or sizing change beat it out of sample in the base scenario, so I didn't ship one.
- **v2's gains are in what the CSV simulator can't credit:**
  - safe operation on a shared account;
  - the final-report snipe;
  - an announcement response instead of a 120 s blackout;
  - an ETF arbitrage path;
  - a markout guard that shrinks sizes when the market is crowded.
- These are supported by testnet evidence, not by backtest dollars.
- **Whether we beat strong teams depends on crowding:** strongly positive at 0–25% crowding, roughly flat at 50%.
- Live code agrees with the backtest: `replay.py` (Engine + Policy) on seasons 1–12 gave a mean of $9.3k against the simulator's $7.4k, with the same sign season by season.

## 5. Failure analysis

- **Crowded market.** If competitors price the same storm information within about 4 s, the core edge disappears (conservative and harsh rows). The markout guard reduces size per symbol but can't create an edge.
- **Confident wrong forecast.** Worst test seasons were −$16k (base) and −$28k (conservative): a storm that turned or an erratic pattern, with every symbol positioned the same way. The allocator halves this tail at a large cost in mean; switch with `control.py size 0.5`, or stop trading if round 1 goes badly.
- **Data gaps.** A late start or a WebSocket outage makes `complete` False, and the bot stops trading for the rest of that round. That costs opportunity, not money.
- **Teammate conflict.** Two directional bots on the same symbols will trade against each other. Split symbols with `--symbols`.
- **Rate starvation.** Teammates using the team's 180 actions per minute delay our orders. The rate file keeps 15% of our share for urgent actions such as the final report and the kill switch.
- **Announcement misparse.** An auto-priced rule fires on a misleading headline. Every rule is logged and alerted; undo with `control.py reset-adjustments`.
- **Stale or empty books.** Quotes older than 3 s are refreshed over REST; empty or one-sided books are never traded (11% of running-phase snapshots had an empty side).

## 6. Next bottleneck

**Measuring the competition: how much of our edge other teams' bots price in, and how fast.** Every other uncertainty is small next to the gap between $7k (base) and $0 (conservative).

To pursue it, we need a testnet or live session where several teams trade, recorded with `tools/recorder.py`, plus our own fills. From that we can estimate:
- per-symbol markouts at +5, +30 and +120 s;
- how quickly book mids converge to our fair value after each storm observation.

That would tell us whether to run v1 at full size, shift toward the scenario allocator, or concentrate on HAVEN, ARCH and PORT, the symbols whose edge survived the conservative scenario.
