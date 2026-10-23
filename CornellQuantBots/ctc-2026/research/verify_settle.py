import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/hv2'); import simmodel as M
z=np.load('/home/claude/v2/research/lib_all.npz'); meta=z['meta']; Cc=z['Ccum'][:,-1].astype(float)
man=json.load(open('/home/claude/data/hurricane-season-research/history/manifest.json'))
H='/home/claude/data/hurricane-season-research/history/'
rows=[]
for s in man['seasons']:
    D=np.zeros(5); CW=np.zeros(5); X=0; L=0
    for st in s['storms']:
        d=np.array([st['damage_pct'][c] for c in M.PHYS]); i=np.where((meta[:,0]==s['season'])&(meta[:,1]==st['round']))[0][0]
        D+=d; CW+=Cc[i]@M.CM; X+=d@M.RECON_COEF; L+=d@M.LOSS_COEF
    S=M.settle(D,CW,X,L)[0]; se=json.load(open(H+s['dir']+'/settlement.json'))
    rows.append([s['season']]+list(S-np.array([se[k] for k in M.SYMS]))+[abs(L-s['season_loss_musd'])])
R=pd.DataFrame(rows,columns=['season']+M.SYMS+['Lerr'])
for name,g in [('train 201-600',R[R.season>200]),('val 101-200',R[(R.season>100)&(R.season<=200)]),('test 1-100',R[R.season<=100])]:
    print(name,' '.join(f"{c}:{g[c].abs().mean():.3f}/{g[c].abs().max():.2f}" for c in M.SYMS+['Lerr']))
print('bias test', R[R.season<=100][M.SYMS].mean().round(3).to_dict())
