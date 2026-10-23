"""ETF basket arbitrage on executable prices (full-depth books).

Opportunity A (sell ETF / buy basket):  revenue(sell q ETF into bids) - cost(buy q*w_i of each component from asks)
Opportunity B (buy ETF / sell basket):  revenue(sell basket into bids)  - cost(buy q ETF from asks)
Both legs are held to settlement, where the ETF settles exactly at its basket value, so a fully executed pair
is riskless. Create/redeem (zero fee, booked at marks) only converts a matched pair into nothing and frees
position-limit room; it is DISABLED by default because its booking semantics are unverified on the testnet.
Testnet evidence (rec1+rec2, 252 snapshots, read-only): ARCH profitable in 27% of snapshots, mean $11.6 when
profitable, episodes last ~10 s; REBLD rarely (6%, ~$1.7). Expect hundreds of dollars per season, not thousands.
"""
import numpy as np
BASKETS = {'ARCH': {'PORT': 1, 'ISLE': 1, 'AQUA': 2, 'GRID': 1, 'FERRY': 2, 'HAVEN': 1},
           'REBLD': {'RECON': 2, 'MAT': 2, 'GEN': 1}}


def walk(levels, q):
    """(total cost, worst price) to trade q against price levels, or (None, None) if not enough depth"""
    got = 0; cost = 0.0; worst = None
    for l in levels:
        t = min(l['quantity'], q - got)
        if t <= 0:
            break
        got += t; cost += t * l['price']; worst = l['price']
        if got >= q:
            return cost, worst
    return None, None


def best_arb(etf, books, max_q=50, min_unit=0.15, min_total=5.0, room=None):
    """books: sym -> (bids, asks) full depth. Returns dict(direction, q, profit, legs=[(sym, side, qty, limit_px)]) or None"""
    W = BASKETS[etf]
    if etf not in books or any(s not in books for s in W):
        return None
    best = None
    for q in (1, 2, 5, 10, 15, 20, 30, 40, 50):
        if q > max_q:
            break
        for direction in ('sell_etf', 'buy_etf'):
            if direction == 'sell_etf':
                e_rev, e_px = walk(books[etf][0], q)
                legs = [(s, 'buy') + walk(books[s][1], q * w) for s, w in W.items()]
                if e_rev is None or any(l[2] is None for l in legs):
                    continue
                profit = e_rev - sum(l[2] for l in legs)
                orders = [(etf, 'sell', q, e_px)] + [(s, 'buy', q * W[s], px) for s, _, c, px in legs]
            else:
                e_cost, e_px = walk(books[etf][1], q)
                legs = [(s, 'sell') + walk(books[s][0], q * w) for s, w in W.items()]
                if e_cost is None or any(l[2] is None for l in legs):
                    continue
                profit = sum(l[2] for l in legs) - e_cost
                orders = [(etf, 'buy', q, e_px)] + [(s, 'sell', q * W[s], px) for s, _, c, px in legs]
            if room is not None and any(room(s, side) < qq for s, side, qq, _ in orders):
                continue
            if profit >= max(min_total, min_unit * q) and (best is None or profit > best['profit']):
                best = dict(etf=etf, direction=direction, q=q, profit=profit, legs=orders)
    return best
