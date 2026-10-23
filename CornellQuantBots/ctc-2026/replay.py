"""Offline replay of a research season through the LIVE code path (Engine + Policy), with a simulated exchange.

  python replay.py --seasons 1-5 --scenario base [--every 5]

Differences from the research backtester (bt/sim.py): this drives the real Engine with feed messages in seq
order (so it also tests message handling), and calls Policy.targets exactly as bot.py does. Fills: aggressive
part against a ladder around the house consensus (market.csv) +/- scenario spread; resting part fills only if
the market later trades through it. Use it to check that live logic and backtest agree.
"""
import argparse, json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M
from library import Library
from engine import Engine
from strategy import Policy, SYMS

SPREAD0 = np.array([.13, .13, .07, .11, .06, .15, .10, .11, .10, .23, .72, .60])
TOP0 = np.array([20, 30, 50, 33, 80, 50, 56, 49, 24, 14, 18, 39], float)
SCEN = {'optimistic': (1.0, 1.0, 1.5, 0.0, 0.0), 'base': (2.0, 1.0, 1.0, 0.5, 0.25), 'conservative': (3.5, 1.5, 0.6, 1.0, 0.5)}


def row_of(t):
    h, m = int(t[11:13]), int(t[14:16]); s = int(t[17:19]) if len(t) > 17 else 0
    return int(((h * 60 + m - 570) * 60 + s) / 60)   # one csv row per fictional minute (= 2 real s)


def run(season, root, lib, scen='base', every=5):
    sp_storm, sp_calm, depth, impact, crowd = SCEN[scen]
    H = os.path.join(root, 'history', f'season-{season:03d}')
    feed = [json.loads(l) for l in open(os.path.join(H, 'feed.jsonl'))]
    c = pd.read_csv(os.path.join(H, 'market.csv'))
    se = json.load(open(os.path.join(H, 'settlement.json'))); settle = np.array([se[s] for s in SYMS])
    eng = Engine(lib); eng.exclude_season = season; pol = Policy()
    pos = np.zeros(12); cash = 0.0; n_orders = 0
    k_imp = impact * (SPREAD0 / 2) / TOP0
    msgs = {}
    for m in feed:
        msgs.setdefault((m['round'], row_of(m['time'])), []).append(m)
    for rnd in (1, 2, 3):
        csv = c[c['round'] == rnd][SYMS].values.astype(float)
        for row in range(390):
            urgent = False
            for m in msgs.get((rnd, row), []):
                ev = eng.on_message(m); urgent |= ev in ('final', 'prelim')
            if row % every and not urgent:
                continue
            f, S, info = eng.fair()
            hs_mult = sp_storm if 30 <= row <= 285 else sp_calm
            hs = SPREAD0 / 2 * hs_mult
            mid = csv[row] + crowd * (f - csv[row]) + k_imp * pos
            orders = pol.targets(f, info, dict(zip(SYMS, mid)), dict(zip(SYMS, hs)), dict(zip(SYMS, pos)),
                                 urgent=urgent and info.get('phase') == 'final')
            exe = min(row + 1, 389)
            for s, side, qty, px, prio, val in orders:
                j = SYMS.index(s); sg = 1 if side == 'buy' else -1; n_orders += 1
                m_exe = csv[exe][j] + crowd * (f[j] - csv[exe][j]) + k_imp[j] * pos[j]
                filled = 0
                for o, sz in zip(hs[j] * np.array([1, 2.5, 4, 8, 16]), TOP0[j] * depth * np.array([1, 1, 2, 3, 10])):
                    lvl = m_exe + sg * o
                    if (sg > 0 and lvl > px) or (sg < 0 and lvl < px):
                        break
                    t = min(sz, qty - filled); filled += t; cash -= sg * t * lvl; pos[j] += sg * t
                    if filled >= qty:
                        break
    return dict(season=season, pnl=cash + pos @ settle, orders=n_orders)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', default='1-3'); ap.add_argument('--scenario', default='base'); ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--root', default=os.path.join(HERE, '..', 'hurricane-season-research'))
    ap.add_argument('--lib', default=os.path.join(HERE, 'data', 'lib_train.npz'))
    a = ap.parse_args(); lo, hi = (int(x) for x in a.seasons.split('-'))
    lib = Library(a.lib); out = [run(s, a.root, lib, a.scenario, a.every) for s in range(lo, hi + 1)]
    df = pd.DataFrame(out); print(df.to_string()); print('mean', round(df.pnl.mean()), 'median', round(df.pnl.median()), 'win', round((df.pnl > 0).mean(), 2))
