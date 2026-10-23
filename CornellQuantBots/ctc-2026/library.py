"""Analog storm library built from the research seasons (restricted to a chosen season set)."""
import json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M
PATTERNS = ['direct', 'chain', 'recurve', 'stall', 'eyewall', 'erratic']


def build(research_root, out_path, seasons=None):
    H = os.path.join(research_root, 'history'); man = json.load(open(os.path.join(H, 'manifest.json')))
    rows, A, OFF = [], [], []
    for s in man['seasons']:
        if seasons is not None and s['season'] not in seasons:
            continue
        obs = {}; adv = {}
        for line in open(os.path.join(H, s['dir'], 'feed.jsonl')):
            m = json.loads(line)
            if m['type'] == 'storm':
                obs.setdefault(m['round'], {})[m['age']] = [m['x'], m['y'], m['vx'], m['vy'], m['radius'], m['intensity']]
            elif m['type'] == 'advisory' and m['source'] == 'official':
                adv.setdefault(m['round'], []).append((m['age'], m['outlook']))
        for st in s['storms']:
            r = st['round']; A.append(np.array([obs[r][a] for a in range(0, 481, 2)], float))
            o = np.full(241, -1, np.int8)
            for a, ol in sorted(adv.get(r, [])):
                o[int(np.ceil(a / 2)):] = PATTERNS.index(ol)
            OFF.append(o); rows.append((s['season'], r, PATTERNS.index(st['pattern'])))
    A = np.array(A); OFF = np.array(OFF); rows = np.array(rows)
    Fc = np.zeros((len(A), 241, 17), np.float32); Cc = np.zeros((len(A), 241, 17), np.float32)
    for i in range(len(A)):
        x, y, vx, vy, r, I = A[i].T
        F, C = M.exposure_steps(*map(M.upsample, (x, y, r, I)))
        Fc[i] = np.cumsum(F, 0)[0::2]; Cc[i] = np.cumsum(C, 0)[0::2]
    np.savez_compressed(out_path, A=A, Fcum=Fc, Ccum=Cc, OFF=OFF, meta=rows)
    return len(A)


class Library:
    def __init__(self, path):
        z = np.load(path)
        self.A = z['A']; self.Fcum = z['Fcum'].astype(float); self.Ccum = z['Ccum'].astype(float); self.OFF = z['OFF']
        self.season = z['meta'][:, 0]; self.round = z['meta'][:, 1]; self.pattern = z['meta'][:, 2]
        self.CWfull = self.Ccum[:, -1] @ M.CM
        self._feat = {}

    def feats(self, k):
        if k not in self._feat:
            self._feat[k] = np.array([smooth_state(a, k) for a in self.A])
        return self._feat[k]


def smooth_state(arr, k):
    k0 = max(0, k - 5); x, y = arr[k, 0], arr[k, 1]
    if k > k0:
        vx = (arr[k, 0] - arr[k0, 0]) / (2 * (k - k0)); vy = (arr[k, 1] - arr[k0, 1]) / (2 * (k - k0))
    else:
        vx, vy = arr[k, 2], arr[k, 3]
    return np.array([x, y, vx, vy, arr[max(0, k - 3):k + 1, 4].mean(), arr[max(0, k - 3):k + 1, 5].mean()])


if __name__ == '__main__':
    root = sys.argv[1]; out = sys.argv[2]
    lo, hi = (int(v) for v in sys.argv[3].split('-')) if len(sys.argv) > 3 else (1, 600)
    print('built', build(root, out, set(range(lo, hi + 1))))
