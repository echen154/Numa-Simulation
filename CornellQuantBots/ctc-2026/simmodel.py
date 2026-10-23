"""Recovered simulator formulas (deterministic part) and the fitted stochastic parameters."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = json.load(open(os.path.join(HERE, 'data', 'case.json')))
P = json.load(open(os.path.join(HERE, 'data', 'params.json')))
PHYS = ['PORT', 'ISLE', 'AQUA', 'GRID', 'FERRY']
SYMS = PHYS + ['RECON', 'MAT', 'GEN', 'HAVEN', 'REIN', 'ARCH', 'REBLD']
LIMITS = CASE['limits']
SITES = [(c, i, s['island'], s['x'], s['y'], s['weight']) for c in PHYS for i, s in enumerate(CASE['companies'][c]['sites'])]
SX = np.array([s[3] for s in SITES], float); SY = np.array([s[4] for s in SITES], float)
SW = np.array([s[5] for s in SITES]); SC = np.array([PHYS.index(s[0]) for s in SITES])
CM = np.zeros((17, 5)); CM[np.arange(17), SC] = SW          # site -> company weight matrix
Q = P['Q']; CLOSE_T = P['close_T']; WIND_K = P['wind_k']; RATES = np.array(P['rates'])
LOSS_COEF = np.array([9, 6, 2.5, 7, 1.2])     # $M insured loss per damage point (exact)
RECON_COEF = np.array([9, 0, 0, 7, 1.2])      # $M of RECON repair work per damage point (exact)
BASE = np.array([100, 60, 35, 50, 25.]); DCOEF = np.array([0.55, 0.38, 0.20, 0.24, 0.05]); CCOEF = np.array([0.02, 0, 0, 0.05, 0.06])
HAVEN_ATT, HAVEN_EXH, REIN_ATT = 400.0, 1300.0, 1300.0
# runtime adjustments set by announcements / human control (per-share additive, per-symbol multiplicative)
ADJ_ADD = np.zeros(10); ADJ_MUL = np.ones(10)
ARCH_W = np.array([1, 1, 2, 1, 2, 0, 0, 0, 1, 0])    # over the 10 stocks
REBLD_W = np.array([0, 0, 0, 0, 0, 2, 2, 1, 0, 0])


def upsample(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return a
    out = np.empty(2 * len(a) - 1); out[0::2] = a; out[1::2] = (a[1:] + a[:-1]) / 2
    return out


def wind(x, y, r, I):
    d = np.hypot(np.asarray(x)[:, None] - SX[None], np.asarray(y)[:, None] - SY[None]) / np.asarray(r)[:, None]
    return np.clip(np.asarray(I), 0, None)[:, None] * np.exp(-WIND_K * d ** 2)


def exposure_steps(x, y, r, I):
    """per-1s-step damage exposure (T,17) and closed indicator (T,17) from 1s arrays"""
    W = wind(x, y, r, I); I_ = np.clip(np.asarray(I), 0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.where(I_[:, None] > 0, W * (I_ ** (Q - 1))[:, None], 0.0)
    return F, (W > CLOSE_T).astype(float)


def exposure(x, y, r, I):
    """from 2s observations: total exposure and closed seconds per site"""
    F, C = exposure_steps(*map(upsample, (x, y, r, I)))
    return F.sum(0), C.sum(0)


def company_close(cl):
    return np.asarray(cl) @ CM


def settle(D, CW, X, L):
    D = np.atleast_2d(D); CW = np.atleast_2d(CW); X = np.atleast_1d(X); L = np.atleast_1d(L)
    phys = BASE - DCOEF * D - CCOEF * CW
    recon = 40 + 24 * (1 - np.exp(-X / 1000)); mat = 28 + 14 * (1 - np.exp(-L / 1200)); gen = 32 + 18 * (1 - np.exp(-CW[:, 3] / 120))
    haven = 75 - 0.06 * np.clip(L - HAVEN_ATT, 0, HAVEN_EXH - HAVEN_ATT); rein = 90 - 0.035 * np.clip(L - REIN_ATT, 0, None)
    out = np.c_[phys, recon, mat, gen, haven, rein] * ADJ_MUL + ADJ_ADD
    return np.c_[out, out @ ARCH_W, out @ REBLD_W]
