"""Storm forecasting: pattern posterior from advisories + analog path futures."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M
from library import smooth_state, PATTERNS

PRIOR = np.log(np.array([M.P['pattern_prior'][p] for p in PATTERNS]) + 1e-6)
LO = np.log(M.P['acc_official'] / ((1 - M.P['acc_official']) / 5))
LW = np.log(M.P['acc_wire'] / ((1 - M.P['acc_wire']) / 5))
SCALE = np.array([60, 60, 0.6, 0.6, 15, 0.15])


def posterior(advs, use_wire=True):
    sc = PRIOR.copy()
    for src, ol in advs:
        if src == 'official':
            sc[PATTERNS.index(ol)] += LO
        elif use_wire:
            sc[PATTERNS.index(ol)] += LW
    p = np.exp(sc - sc.max()); return p / p.sum()


def select_analogs(lib, state, k, post, K=40, exclude_season=None):
    Fe = lib.feats(k); dist = (((Fe - state) / SCALE) ** 2).sum(1)
    if exclude_season is not None:
        dist = np.where(lib.season == exclude_season, np.inf, dist)
    alloc = np.floor(post * K + 0.5).astype(int); idx = []
    for p in range(6):
        if alloc[p] == 0:
            continue
        cand = np.where(lib.pattern == p)[0]
        idx += list(cand[np.argsort(dist[cand])[:alloc[p]]])
    return np.array(idx, int)


def analog_paths(lib, state, k, idx):
    """shifted/scaled future 2s paths (n, 241-k, 4): x,y,r,I"""
    out = []
    for j in idx:
        fut = lib.A[j, k:]
        dx = state[0] - lib.A[j, k, 0]; dy = state[1] - lib.A[j, k, 1]
        rs = state[4] / max(lib.A[j, max(0, k - 3):k + 1, 4].mean(), 1)
        Is = state[5] / max(lib.A[j, max(0, k - 3):k + 1, 5].mean(), 1e-3)
        out.append(np.c_[fut[:, 0] + dx, fut[:, 1] + dy, fut[:, 4] * rs, np.clip(fut[:, 5] * Is, 0, None)])
    return np.array(out)


def path_exposure(paths):
    """future exposure (n,17) and closed seconds (n,17) excluding the current point"""
    n = len(paths); F = np.zeros((n, 17)); C = np.zeros((n, 17))
    if paths.shape[1] < 2:
        return F, C
    for i, p in enumerate(paths):
        x, y, r, I = (M.upsample(p[:, c]) for c in range(4))
        Fs, Cs = M.exposure_steps(x[1:], y[1:], r[1:], I[1:])
        F[i] = Fs.sum(0); C[i] = Cs.sum(0)
    return F, C


def futures(lib, obs, k, advs, K=40, exclude_season=None):
    """obs: filled (k+1,6) array; advs: [(source, outlook)] seen so far"""
    if k >= 240:
        return np.zeros((1, 17)), np.zeros((1, 17))
    state = smooth_state(obs, k); post = posterior(advs)
    idx = select_analogs(lib, state, k, post, K, exclude_season)
    return path_exposure(analog_paths(lib, state, k, idx))


def cone_adjust(paths, k, cone, scale_spread=True):
    """Shift the analog ensemble so its mean passes through the official cone points (ages in s);
    optionally rescale member deviations to the cone spread."""
    if not cone:
        return paths
    n, T, _ = paths.shape
    ages = 2 * (k + np.arange(T))
    mean = paths[:, :, :2].mean(0)
    pts = [(c['age'], c['x'], c['y'], c['spread']) for c in cone if c['age'] > 2 * k and c['age'] <= 480]
    if not pts:
        return paths
    ca = np.array([2 * k] + [p[0] for p in pts], float)
    corr_x = np.array([0.0] + [p[1] - mean[min(T - 1, (p[0] - 2 * k) // 2), 0] for p in pts])
    corr_y = np.array([0.0] + [p[2] - mean[min(T - 1, (p[0] - 2 * k) // 2), 1] for p in pts])
    cx = np.interp(ages, ca, corr_x); cy = np.interp(ages, ca, corr_y)
    out = paths.copy(); out[:, :, 0] += cx; out[:, :, 1] += cy
    if scale_spread and n > 2:
        sp = np.interp(ages, ca, np.array([0.0] + [p[3] for p in pts]))
        newmean = out[:, :, :2].mean(0)
        dev = out[:, :, :2] - newmean
        cur = np.sqrt((dev ** 2).sum(-1).mean(0)) + 1e-6
        fac = np.clip(sp / cur, 0.3, 3.0)
        out[:, :, :2] = newmean + dev * fac[None, :, None]
    return out


def futures_cone(lib, obs, k, advs, cone, K=40, exclude_season=None, scale_spread=True):
    if k >= 240:
        return np.zeros((1, 17)), np.zeros((1, 17))
    state = smooth_state(obs, k); post = posterior(advs)
    idx = select_analogs(lib, state, k, post, K, exclude_season)
    P = analog_paths(lib, state, k, idx)
    return path_exposure(cone_adjust(P, k, cone, scale_spread))
