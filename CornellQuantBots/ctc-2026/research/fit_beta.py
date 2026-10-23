import sys, numpy as np, pandas as pd, json
sys.path.insert(0,'/home/claude/v2/bt'); import runner, simmodel as M
data=runner.load(101,150)
rows=[]
for s,(L,csv,se) in data.items():
    for d in L:
        h=csv[d['round']][min(d['row']+1,389)]
        rows.append((d['round'],d['row'],d['fc']-h,d['f0']-h,se-h))
R=pd.DataFrame(rows,columns=['round','row','xc','x0','y'])
def bucket(row):
    if row<30: return 'pre'
    if row<=270:
        a=2*(row-30); return f'storm{min(a//60*60,420):03d}'
    if row<286: return 'after'
    if row<371: return 'prelim'
    return 'final'
R['b']=R.row.map(bucket)
out={}
for (rnd,b),g in R.groupby(['round','b']):
    X=np.stack(g.xc.values); Y=np.stack(g.y.values); X0=np.stack(g.x0.values)
    bs=(X*Y).sum(0)/np.maximum((X*X).sum(0),1e-9); bp=(X[:,:10]*Y[:,:10]).sum()/(X[:,:10]**2).sum()
    b0=(X0[:,:10]*Y[:,:10]).sum()/(X0[:,:10]**2).sum()
    out[(rnd,b)]=bs
    print(rnd,b,'pooled beta Fc %.2f  F0 %.2f'%(bp,b0),' '.join(f'{s}:{v:+.2f}' for s,v in zip(M.SYMS,bs)))
pd.to_pickle(out,'beta_v1fit.pkl')
