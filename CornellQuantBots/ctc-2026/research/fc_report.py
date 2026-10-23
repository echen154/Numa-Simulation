import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/hv2'); import simmodel as M
rows=sum([pd.read_pickle(f) for f in sys.argv[1].split(',')],[])
names=sys.argv[2].split(',')
S=['PORT','ISLE','AQUA','GRID','FERRY','RECON','MAT','GEN','HAVEN','REIN']; ix=[M.SYMS.index(s) for s in S]
ages=sorted(set(r['age'] for r in rows))
print('MAE (sum over 10 stocks, $) and beta of (settle-house) on (fair-house), pooled')
for a in ages[::2]:
    R=[r for r in rows if r['age']==a]
    st=np.array([r['set'] for r in R])[:,ix]; h=np.array([r['house'] for r in R])[:,ix]
    line=f'age {a:3d} house {np.abs(h-st).mean(0).sum():6.2f}'
    for n in names:
        f=np.array([r[n] for r in R])[:,ix]
        x=(f-h).ravel(); y=(st-h).ravel(); b=np.dot(x,y)/np.dot(x,x)
        line+=f' | {n} {np.abs(f-st).mean(0).sum():6.2f} b={b:+.2f}'
    print(line)
