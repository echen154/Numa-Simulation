import sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/v2/bt'); sys.path.insert(0, '/home/claude/v2/hv2')
import sim, simmodel as M
H = '/home/claude/data/hurricane-season-research/history/'


def load(lo, hi):
    rows = []
    for f in glob.glob('/home/claude/v2/bt/pc_*.pkl'):
        a, b = (int(x) for x in f.split('pc_')[1].split('.pkl')[0].split('_'))
        if b < lo or a > hi:
            continue
        rows += [d for d in pd.read_pickle(f) if lo <= d['season'] <= hi]
    by = {}
    for d in rows:
        by.setdefault(d['season'], []).append(d)
    data = {}
    for s, L in by.items():
        c = pd.read_csv(H + f'season-{s:03d}/market.csv')
        csv = {r: g[M.SYMS].values.astype(float) for r, g in c.groupby('round')}
        se = json.load(open(H + f'season-{s:03d}/settlement.json'))
        data[s] = (L, csv, np.array([se[k] for k in M.SYMS]))
    return data


def evaluate(data, policy, scen='base', seasons=None):
    out = []
    for s, (L, csv, se) in sorted(data.items()):
        if seasons is not None and s not in seasons:
            continue
        r = sim.run_season(L, csv, se, policy, scen)
        out.append(dict(season=s, pnl=r['pnl'], actions=r['actions'], traded=r['traded'], cost=r['cost'], **{f'p_{k}': v for k, v in zip(M.SYMS, r['pnl_sym'])}))
    return pd.DataFrame(out)


def summary(df, name=''):
    p = df.pnl.values
    return dict(name=name, n=len(p), mean=round(p.mean()), median=round(np.median(p)), win=round((p > 0).mean(), 3),
                p10=round(np.percentile(p, 10)), p5=round(np.percentile(p, 5)), worst=round(p.min()), best=round(p.max()),
                sd=round(p.std()), actions=round(df.actions.mean()), traded_k=round(df.traded.mean() / 1000), cost=round(df.cost.mean()))
