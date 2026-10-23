"""Feed-driven state and fair value (v2). Call on_message(m) for each feed message; fair() for valuation."""
import math, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M, forecast as FC, valuation as V

N_OBS = 241


def _num(v, lo, hi):
    return isinstance(v, (int, float)) and math.isfinite(v) and lo <= v <= hi


class StormState:
    def __init__(self, rnd, name):
        self.round = rnd; self.name = name
        self.obs = np.full((N_OBS, 6), np.nan); self.advs = []; self.cones = []
        self.prelim = None; self.final = None; self.dissipated = False

    def add_obs(self, m):
        if not _num(m.get('age'), 0, 480):
            return False
        k = int(m['age']) // 2
        vals = [m.get('x'), m.get('y'), m.get('vx', 0.0), m.get('vy', 0.0), m.get('radius'), m.get('intensity')]
        if not (0 <= k < N_OBS and _num(vals[0], -500, 1500) and _num(vals[1], -500, 1500) and _num(vals[4], 1, 1000) and _num(vals[5], 0, 10)):
            return False
        self.obs[k] = [float(v) if _num(v, -1e6, 1e6) else 0.0 for v in vals]
        return True

    def add_track(self, pts):
        for p in pts or []:
            if len(p) < 4:
                continue
            a, x, y, r = p[:4]
            if not _num(a, 0, 480):
                continue
            k = int(a) // 2
            if 0 <= k < N_OBS and np.isnan(self.obs[k, 0]) and _num(x, -500, 1500) and _num(y, -500, 1500) and _num(r, 1, 1000):
                self.obs[k, 0] = x; self.obs[k, 1] = y; self.obs[k, 4] = r

    def last_k(self):
        ok = np.where(~np.isnan(self.obs[:, 0]))[0]
        return int(ok.max()) if len(ok) else -1

    def filled(self):
        last = self.last_k()
        if last < 0:
            return None
        a = self.obs[:last + 1].copy()
        for c in range(6):
            col = a[:, c]; ok = ~np.isnan(col)
            if ok.sum() == 0:
                col[:] = 1.0 if c == 5 else 0.0
            elif (~ok).any():
                col[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], col[ok])
        return a

    def coverage(self):
        k = self.last_k()
        if k < 0:
            return 1.0
        if np.isnan(self.obs[0, 0]):
            return 0.0
        return float((~np.isnan(self.obs[:k + 1, 0])).mean())

    def realized(self):
        a = self.filled()
        if a is None:
            return np.zeros(17), np.zeros(17)
        x, y, vx, vy, r, I = a.T
        return M.exposure(x, y, r, I)


class Engine:
    def __init__(self, lib):
        self.lib = lib; self.exclude_season = None; self.use_cone = False; self.reset()

    def reset(self):
        self.season = V.SeasonState(); self.storms = {}; self.final_done = set(); self.events = []

    def current_round(self):
        return max(self.storms) if self.storms else 0

    def on_message(self, m):
        """returns an event name for messages that should trigger immediate action"""
        t = m.get('type'); rnd = m.get('round')
        if t == 'snapshot':
            for d in m.get('damage') or []:
                self.on_message(d)
            if m.get('track') and self.storms.get(rnd):
                self.storms[rnd].add_track(m['track'])
            return None
        if t in ('announcement', 'news'):
            return t
        if not isinstance(rnd, int) or not 1 <= rnd <= 3:
            return None
        if t == 'storm_formed' and rnd == 1 and any(r > 1 for r in self.storms):
            self.reset()   # a new season on the same connection (testnet)
        st = self.storms.get(rnd)
        if st is None:
            st = self.storms[rnd] = StormState(rnd, m.get('storm'))
        if t == 'storm':
            st.add_obs(m)
        elif t == 'track':
            st.add_track(m.get('points'))
        elif t == 'advisory':
            if m.get('outlook') in FC.PATTERNS and m.get('source') in ('official', 'wire'):
                st.advs.append((m['source'], m['outlook'], m.get('age', 0)))
                if m['source'] == 'official' and m.get('cone'):
                    st.cones.append((m.get('age', 0), m['cone']))
        elif t == 'storm_dissipated':
            st.dissipated = True
        elif t == 'damage_report':
            dp = m.get('damage_pct') or {}
            if not all(_num(dp.get(c), 0, 100) for c in M.PHYS):
                return None
            d = np.array([dp[c] for c in M.PHYS], float)
            if m.get('stage') == 'preliminary':
                st.prelim = d; return 'prelim'
            if m.get('stage') == 'final' and rnd not in self.final_done:
                st.final = d; Fr, Cr = st.realized(); self.season.add_final(d, Fr, Cr); self.final_done.add(rnd)
                return 'final'
        return t

    def complete(self):
        rnd = self.current_round()
        for r in range(1, rnd):
            if r not in self.final_done or self.storms.get(r) is None or self.storms[r].coverage() < 0.95:
                return False
        return rnd == 0 or self.storms[rnd].coverage() >= 0.9

    def fair(self, K=40, n_noise=4):
        """-> (mean (12,), scenario matrix (N,12), info)"""
        rnd = self.current_round(); comp = self.complete()
        if rnd == 0:
            S = V.scenarios(self.season, self.lib, 3, np.zeros(17), np.zeros(17), np.zeros((1, 17)), np.zeros((1, 17)), n_noise=160, exclude_season=self.exclude_season)
            return S.mean(0), S, dict(round=0, age=-1, phase='pre', complete=comp)
        st = self.storms[rnd]; left = 3 - rnd; k = st.last_k()
        if rnd in self.final_done:
            S = V.scenarios(self.season, self.lib, left, np.zeros(17), np.zeros(17), np.zeros((1, 17)), np.zeros((1, 17)), n_noise=1, n_future=160 if left else 1, exclude_season=self.exclude_season)
            return S.mean(0), S, dict(round=rnd, age=480, phase='final', complete=comp)
        if k < 0:
            S = V.scenarios(self.season, self.lib, left + 1, np.zeros(17), np.zeros(17), np.zeros((1, 17)), np.zeros((1, 17)), n_noise=160, exclude_season=self.exclude_season)
            return S.mean(0), S, dict(round=rnd, age=-1, phase='pre', complete=comp)
        Fr, Cr = st.realized()
        if k >= 240 or st.dissipated:
            S = V.scenarios(self.season, self.lib, left, Fr, Cr, np.zeros((1, 17)), np.zeros((1, 17)), n_noise=160, prelim=st.prelim, exclude_season=self.exclude_season)
            return S.mean(0), S, dict(round=rnd, age=2 * k, phase='prelim' if st.prelim is not None else 'after', complete=comp)
        a = st.filled(); advs = [(s_, o_) for s_, o_, ag in st.advs if ag <= 2 * k]
        if self.use_cone and st.cones:
            cone = [c for ag, c in st.cones if ag <= 2 * k]
            fF, fC = FC.futures_cone(self.lib, a, k, advs, cone[-1] if cone else None, K, self.exclude_season)
        else:
            fF, fC = FC.futures(self.lib, a, k, advs, K, self.exclude_season)
        S = V.scenarios(self.season, self.lib, left, Fr, Cr, fF, fC, n_noise=n_noise, exclude_season=self.exclude_season)
        return S.mean(0), S, dict(round=rnd, age=2 * k, phase='storm', complete=comp)

    def exact_after_final(self):
        """deterministic settlement contribution once the current round's final report is in (round 3: exact)."""
        return self.fair()
