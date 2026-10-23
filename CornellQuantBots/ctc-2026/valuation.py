"""Scenario valuation: joint Monte Carlo distribution of all 12 settlement values.

Scenarios = (current-storm path futures from the forecaster) x (site damage noise) x (future storms
drawn from the analog library). Common random numbers (fixed seed) keep successive calls smooth.
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M

NOISE = M.P['noise_sd']
PRELIM_W = np.array(M.P['prelim_w'])


class SeasonState:
    """Exact information from finished storms."""
    def __init__(self):
        self.rem = np.ones(17); self.D = np.zeros(5); self.CW = np.zeros(5); self.X = 0.0; self.L = 0.0

    def add_final(self, dmg5, F_site, cl_site):
        exp = self.rem * (1 - np.exp(-M.RATES * F_site))
        cexp = 100 * exp @ M.CM
        scale = np.where(cexp > 1e-9, np.asarray(dmg5) / np.maximum(cexp, 1e-9), 0)
        self.rem = np.clip(self.rem - exp * scale[M.SC], 0, 1)
        self.D = self.D + dmg5; self.CW = self.CW + M.company_close(cl_site)
        self.X += float(np.dot(dmg5, M.RECON_COEF)); self.L += float(np.dot(dmg5, M.LOSS_COEF))

    def copy(self):
        s = SeasonState(); s.rem = self.rem.copy(); s.D = self.D.copy(); s.CW = self.CW.copy(); s.X = self.X; s.L = self.L
        return s


def scenarios(sea, lib, rounds_left, Freal, Creal, futF, futC, n_noise=4, prelim=None, n_future=25, seed=12345,
              exclude_season=None, exact_current=None):
    """Return settlement scenario matrix (N, 12).
    exact_current: final damage pct of current storm if known (then the current storm has no noise)."""
    g = np.random.default_rng(seed)
    if exact_current is not None:
        d = np.atleast_2d(np.asarray(exact_current, float)); frac = None
        Cc = np.atleast_2d(Creal); N = 1
        exp = sea.rem * (1 - np.exp(-M.RATES * Freal)); cexp = 100 * exp @ M.CM
        sc = np.where(cexp > 1e-9, d[0] / np.maximum(cexp, 1e-9), 0); frac = np.atleast_2d(exp * sc[M.SC])
    else:
        K = len(futF); N = K * n_noise
        Fx = np.repeat(Freal[None] + futF, n_noise, 0); Cc = np.repeat(Creal[None] + futC, n_noise, 0)
        eps = np.exp(g.normal(-NOISE ** 2 / 2, NOISE, (N, 17)))
        frac = sea.rem[None] * (1 - np.exp(-M.RATES * eps * Fx))
        d = 100 * frac @ M.CM
        if prelim is not None:
            d = d + PRELIM_W * (np.asarray(prelim) - d.mean(0))
    CWc = Cc @ M.CM
    Mf = n_future if rounds_left > 0 else 1
    D = np.repeat(sea.D + d, Mf, 0); CW = np.repeat(sea.CW + CWc, Mf, 0)
    X = np.repeat(sea.X + d @ M.RECON_COEF, Mf); L = np.repeat(sea.L + d @ M.LOSS_COEF, Mf)
    rem2 = np.repeat(np.clip(sea.rem - frac, 0, 1), Mf, 0)
    if rounds_left > 0:
        pool = np.arange(len(lib.season)) if exclude_season is None else np.where(lib.season != exclude_season)[0]
        NM = len(D)
        for _ in range(rounds_left):
            j = pool[g.integers(0, len(pool), NM)]
            e2 = np.exp(g.normal(-NOISE ** 2 / 2, NOISE, (NM, 17)))
            fr = rem2 * (1 - np.exp(-M.RATES * e2 * lib.Fcum[j, -1])); dj = 100 * fr @ M.CM
            rem2 = np.clip(rem2 - fr, 0, 1)
            D = D + dj; CW = CW + lib.CWfull[j]; X = X + dj @ M.RECON_COEF; L = L + dj @ M.LOSS_COEF
    return M.settle(D, CW, X, L)
