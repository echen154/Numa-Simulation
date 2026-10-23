"""Trading policy (validated in backtests on untouched seasons 1-100):

core   : signal = beta(phase, age) * (fair - mid); target = limit * clip((|signal| - hs) / (2 hs), 0, 1) * sign,
         moved at most 34% of the limit per order, limit price = mid + signal -/+ hs.
         beta: 0 before storm age 120 s, ramps to 0.8 at 240 s, 1.0 after the storm and after the final report.
final  : on a final damage report the current-storm damage is exact; trade immediately (urgent priority)
         with a half-spread margin.
guard  : markout monitor - if our fills keep losing against the mid 30 s later (crowded market), sizes shrink.
Overrides (announcement module / human): per-symbol fair override, per-symbol pause, global size scale.
"""
import time, numpy as np, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M

SYMS = M.SYMS
LIM = np.array([M.LIMITS[s] for s in SYMS], float)
MIN_HS = np.array([.05, .04, .02, .04, .02, .06, .04, .03, .05, .05, .3, .25])


def beta_for(info):
    if not info.get('complete'):
        return 0.0
    ph = info.get('phase')
    if ph == 'final':
        return 1.0
    if ph in ('after', 'prelim'):
        return 1.0
    if ph == 'storm':
        return 0.8 * float(np.clip((info['age'] - 120) / 120, 0, 1))
    return 0.0


class MarkoutGuard:
    """Tracks (fill price, side, mid 30 s later). Size multiplier falls when markouts are persistently negative."""
    def __init__(self, horizon=30.0, floor=0.25):
        self.h = horizon; self.floor = floor; self.pending = []; self.score = {}

    def on_fill(self, sym, side, qty, price):
        self.pending.append((time.time(), sym, side, qty, price))

    def update(self, mids):
        now = time.time(); keep = []
        for t, s, side, q, px in self.pending:
            if now - t >= self.h and mids.get(s):
                mo = (mids[s] - px) * (1 if side == 'buy' else -1)       # $/share, + = good
                self.score[s] = 0.8 * self.score.get(s, 0.0) + 0.2 * mo
            else:
                keep.append((t, s, side, q, px))
        self.pending = keep

    def mult(self, sym, hs):
        sc = self.score.get(sym, 0.0)
        if sc >= -0.5 * hs:
            return 1.0
        return float(max(self.floor, 1.0 + sc / (4 * hs)))


class Policy:
    def __init__(self, mult=2.0, max_clip=0.34, size_scale=1.0):
        self.mult = mult; self.max_clip = max_clip; self.size_scale = size_scale
        self.fair_override = {}     # sym -> (value, expiry)
        self.guard = MarkoutGuard()
        self.announce_until = {}    # sym -> time: trade at beta=1 (house lags ~45 s after an exact announcement)

    def targets(self, fair, info, mids, hss, pos, paused=lambda s: False, urgent=False):
        """returns list of (sym, side, qty, limit_price, priority, edge_value)"""
        b0 = beta_for(info); out = []; now = time.time()
        for j, s in enumerate(SYMS):
            mid = mids.get(s); hs = hss.get(s)
            if mid is None or hs is None or paused(s):
                continue
            hs = max(hs, MIN_HS[j]); f = fair[j]; b = b0
            if self.announce_until.get(s, 0) > now and info.get('complete'):
                b = 1.0
            ov = self.fair_override.get(s)
            if ov and ov[1] > now:
                f = ov[0]; b = 1.0
            if b <= 0:
                continue
            E = mid + b * (f - mid); edge = E - mid
            # identical to the backtested policy: trade toward the target, including back to flat
            scale = 0.0 if abs(edge) <= hs else float(np.clip((abs(edge) - hs) / (self.mult * hs), 0, 1))
            scale *= self.size_scale * self.guard.mult(s, hs)
            tgt = round(LIM[j] * scale * np.sign(edge))
            want = tgt - pos.get(s, 0)
            if want == 0:
                continue
            qty = int(abs(want) if urgent else min(abs(want), max(1, self.max_clip * LIM[j])))
            if qty < max(1, int(0.02 * LIM[j])):
                continue
            side = 'buy' if want > 0 else 'sell'
            margin = hs * (0.5 if urgent else 1.0)
            px = E - margin if side == 'buy' else E + margin
            out.append((s, side, qty, px, 'urgent' if urgent else 'normal', abs(edge) * qty))
        out.sort(key=lambda o: -o[5])
        return out
