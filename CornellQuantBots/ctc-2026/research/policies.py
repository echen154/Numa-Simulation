"""Trading policies for the backtester. Each policy: f(d, mid, hs, pos, round) -> (target (12,), limit price (12,))."""
import numpy as np, sys
sys.path.insert(0, '/home/claude/v2/hv2')
import simmodel as M
LIM = np.array([M.LIMITS[s] for s in M.SYMS], float)
MIN_HS = np.array([.15, .04, .02, .04, .03, .08, .04, .03, .1, .05, .8, .3])


def age_of(row):
    return 2 * (row - 30)


def beta_v1(row):
    if row < 30:
        return 0.0
    if row >= 371:
        return 1.0
    if row > 270:
        return 1.0
    return 0.8 * float(np.clip((age_of(row) - 120) / 120, 0, 1))


def policy_v1(d, mid, hs, pos, r, use='f0', mult=2.0, max_clip=0.34):
    """the original bot: per-symbol, signal-to-spread sizing"""
    b = beta_v1(d['row']); f = d[use].astype(float)
    hs = np.maximum(hs, MIN_HS)
    E = mid + b * (f - mid); edge = E - mid
    scale = np.clip((np.abs(edge) - hs) / (mult * hs), 0, 1)
    tgt = np.round(LIM * scale * np.sign(edge))
    tgt = np.where(np.abs(edge) <= hs, 0, tgt)
    step = np.clip(tgt - pos, -max_clip * LIM, max_clip * LIM)
    tgt = pos + step
    px = np.where(tgt > pos, E - hs, E + hs)
    if b == 0:
        tgt = pos.copy()
    return tgt, px


def policy_settle_simple(d, mid, hs, pos, r):
    """trust the model fully at all times (beta=1), same sizing"""
    f = d['fc'].astype(float); hs = np.maximum(hs, MIN_HS)
    edge = f - mid
    scale = np.clip((np.abs(edge) - hs) / (2 * hs), 0, 1)
    tgt = np.round(LIM * scale * np.sign(edge)); tgt = np.where(np.abs(edge) <= hs, 0, tgt)
    px = np.where(tgt > pos, f - hs, f + hs)
    return tgt, px


class Allocator:
    """Scenario-based expected-PnL allocator.
    maximize  (mu - mid).x - spread*|x - pos| - (lam/2) x' C x   s.t. |x| <= LIM
    mu = mid + beta(row) * (f - mid);  C = cov of model scenarios (correlations across all 12 symbols)."""
    def __init__(self, beta_fn, lam=1e-3, use='fc', cov_infl=1.0, etf=True, trade_frac=1.0, min_edge_mult=1.0, hold_band=0.0):
        self.beta_fn = beta_fn; self.lam = lam; self.use = use; self.cov_infl = cov_infl; self.etf = etf
        self.trade_frac = trade_frac; self.min_edge_mult = min_edge_mult; self.hold_band = hold_band

    def __call__(self, d, mid, hs, pos, r):
        b = self.beta_fn(d['row'], r)
        f = d[self.use].astype(float); hs = np.maximum(hs, MIN_HS)
        if np.all(b == 0):
            return pos.copy(), mid.copy()
        mu = mid + b * (f - mid); g = mu - mid
        S = d['scen'].astype(float)
        C = np.cov(S.T) * self.cov_infl + np.diag((0.05 * hs) ** 2 + 1e-6)
        if not self.etf:
            C[10:, :] = 0; C[:, 10:] = 0; C[10, 10] = C[11, 11] = 1e9
        x = pos.copy()
        # projected coordinate ascent on concave objective with linear cost around current pos
        for _ in range(30):
            for j in range(12):
                cj = C[j, j] * self.lam
                rest = self.lam * (C[j] @ x - C[j, j] * x[j])
                # optimum for buying (x>pos) and selling (x<pos) branches
                xb = (g[j] - hs[j] * self.min_edge_mult - rest) / cj
                xs = (g[j] + hs[j] * self.min_edge_mult - rest) / cj
                if xb > pos[j]:
                    x[j] = min(xb, LIM[j])
                elif xs < pos[j]:
                    x[j] = max(xs, -LIM[j])
                else:
                    x[j] = pos[j]
        tgt = pos + self.trade_frac * (np.round(x) - pos)
        if self.hold_band > 0:
            small = np.abs(tgt - pos) < self.hold_band * LIM
            tgt = np.where(small, pos, tgt)
        px = np.where(tgt > pos, mu - hs, mu + hs)
        return tgt, px
