"""Event-driven execution simulator for the Hurricane Season case.

Uses precomputed point-in-time model outputs (precompute.py) and the historical house consensus
(market.csv) as the 'fair house mid'. Tradable prices are built from execution scenarios:

  mid_eff(t) = csv(t) + crowd * (fc(t_last_decision) - csv(t)) + impact_k * our_position
  bid/ask    = mid_eff -/+ half_spread(phase) ; depth ladder per symbol

Orders: aggressive limit at a price; walks the ladder up to that price; any remainder rests until
the next decision and fills only if the market later trades through it (adverse selection).
Latency: an order decided at row i executes at row i + latency_rows (prices may have moved).
Accounting: cash + position * settlement. Every fill, order and cancel is counted.
"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0, '/home/claude/v2/hv2')
import simmodel as M

SYMS = M.SYMS; NS = len(SYMS)
LIM = np.array([M.LIMITS[s] for s in SYMS], float)
# recorded testnet medians (rec1, quiet phases): full spread and top-of-book size
SPREAD0 = np.array([.13, .13, .07, .11, .06, .15, .10, .11, .10, .23, .72, .60])
TOP0 = np.array([20, 30, 50, 33, 80, 50, 56, 49, 24, 14, 18, 39], float)

SCENARIOS = {
    # spread multiplier in storm / calm, depth multiplier, impact (fraction of half-spread per top-size of position),
    # crowding (share of model edge already in price), latency rows (2 s each), rate cap actions/min
    'optimistic':   dict(sp_storm=1.0, sp_calm=1.0, depth=1.5, impact=0.0, crowd=0.0, lat=1, rate=120),
    'base':         dict(sp_storm=2.0, sp_calm=1.0, depth=1.0, impact=0.5, crowd=0.25, lat=1, rate=100),
    'conservative': dict(sp_storm=3.5, sp_calm=1.5, depth=0.6, impact=1.0, crowd=0.5, lat=2, rate=80),
    'harsh':        dict(sp_storm=5.0, sp_calm=2.0, depth=0.4, impact=1.5, crowd=0.7, lat=3, rate=60),
}


def ladder(j, hs, top, depth_mult):
    """price offsets (from mid) and sizes of the house ladder for symbol j"""
    offs = hs * np.array([1.0, 2.5, 4.0, 8.0, 16.0])
    sizes = top * depth_mult * np.array([1, 1, 2, 3, 10])
    return offs, sizes


class Market:
    def __init__(self, csv_rows, scen):
        self.csv = csv_rows  # (390, 12) house consensus per row
        self.sc = scen

    def half_spread(self, row):
        storm = 30 <= row <= 285
        return SPREAD0 / 2 * (self.sc['sp_storm'] if storm else self.sc['sp_calm'])


def run_season(season_rows, csv_by_round, settle, policy, scen_name='base', record=False):
    """season_rows: list of precomputed decision dicts (rounds 1..3); policy(state)->target positions (12,)"""
    sc = SCENARIOS[scen_name]
    pos = np.zeros(NS); cash = 0.0; cashv = np.zeros(NS); actions = 0; traded = 0.0; cost = 0.0
    log = []
    by_round = {}
    for d in season_rows:
        by_round.setdefault(d['round'], []).append(d)
    impact_k = sc['impact'] * (SPREAD0 / 2) / TOP0          # $ per share of our position
    action_times = []
    for r in (1, 2, 3):
        csv = csv_by_round[r]; mkt = Market(csv, sc)
        decs = sorted(by_round[r], key=lambda d: d['row'])
        resting = {}   # j -> (qty signed, price)
        for n, d in enumerate(decs):
            row = d['row']; exe = min(row + sc['lat'], 389)
            nxt = decs[n + 1]['row'] if n + 1 < len(decs) else 389
            hs = mkt.half_spread(exe)
            fc = 0.5 * (d['fc'].astype(float) + d['f0'].astype(float))   # proxy for competitors' consensus view
            mid_now = csv[row] + sc['crowd'] * (fc - csv[row]) + impact_k * pos     # what we observe
            # cancel old resting
            for j in list(resting):
                actions += 1; action_times.append((r, row)); del resting[j]
            tgt, limit_px = policy(d, mid_now, hs, pos.copy(), r)
            # rate limiting: per decision interval budget
            budget = int(sc['rate'] * (max(nxt - row, 1) * 2) / 60)
            want = tgt - pos
            order = np.argsort(-np.abs(want) * np.maximum(np.abs(limit_px - mid_now), 1e-9))
            mid_exe = csv[exe] + sc['crowd'] * (fc - csv[exe]) + impact_k * pos
            for j in order:
                q = want[j]
                if abs(q) < max(1, 0.02 * LIM[j]) or budget <= 0:
                    continue
                budget -= 1; actions += 1; action_times.append((r, row))
                offs, sizes = ladder(j, hs[j], TOP0[j], sc['depth'])
                side = np.sign(q); px_lim = limit_px[j]
                filled = 0.0; spent = 0.0
                for o, sz in zip(offs, sizes):
                    lvl = mid_exe[j] + side * o
                    if (side > 0 and lvl > px_lim) or (side < 0 and lvl < px_lim):
                        break
                    take = min(sz, abs(q) - filled)
                    filled += take; spent += take * lvl
                    if filled >= abs(q):
                        break
                if filled > 0:
                    pos[j] += side * filled; cash -= side * spent; cashv[j] -= side * spent; traded += filled * abs(mid_exe[j])
                    cost += abs(spent - filled * mid_exe[j])
                    mid_exe = csv[exe] + sc['crowd'] * (fc - csv[exe]) + impact_k * pos
                rem = abs(q) - filled
                if rem >= 1:
                    resting[j] = (side * rem, px_lim)
            # resting orders fill if market trades through them before next decision
            for j, (q, px) in list(resting.items()):
                side = np.sign(q)
                for rr in range(exe + 1, nxt + 1):
                    m = csv[min(rr, 389)][j] + sc['crowd'] * (fc[j] - csv[min(rr, 389)][j]) + impact_k[j] * pos[j]
                    hsj = mkt.half_spread(rr)[j]
                    touched = (m + hsj <= px) if side > 0 else (m - hsj >= px)
                    if touched:
                        f = min(abs(q), TOP0[j] * sc['depth'])
                        pos[j] += side * f; cash -= side * f * px; cashv[j] -= side * f * px
                        q = q - side * f
                        if abs(q) < 1:
                            break
                if abs(q) >= 1:
                    resting[j] = (q, px)
                else:
                    resting.pop(j, None)
            if record:
                log.append((r, row, pos.copy(), cash))
        # orders do not survive the close
        resting.clear()
    pnl_sym = cash * 0  # placeholder
    pnl = cash + pos @ settle
    # per-symbol pnl needs cash per symbol; approximate via final marks: not tracked separately
    per_min = {}
    for rt in action_times:
        per_min[rt] = per_min.get(rt, 0) + 1
    return dict(pnl=pnl, pnl_sym=cashv + pos * settle, actions=actions, traded=traded, cost=cost, final_pos=pos, log=log)
