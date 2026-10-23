"""Precompute point-in-time fair values / scenario samples for backtesting (train-only model).
For season s, round r, decision row i (2 s rows, row 30 = storm age 0): uses only info observable at that row."""
import sys, json, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/claude/v2/hv2')
import simmodel as M, forecast as FC, valuation as V
from library import Library
lib=Library('/home/claude/v2/hv2/data/lib_train.npz')
z=np.load('/home/claude/v2/research/lib_all.npz'); A=z['A']; meta=z['meta']; Fc=z['Fcum'].astype(float); Cc=z['Ccum'].astype(float)
feed=pd.read_pickle('/home/claude/work/feed.pkl')
lo,hi=int(sys.argv[1]),int(sys.argv[2]); out=sys.argv[3]
ADV={}; FIN={}; PRE={}
for f in feed:
    if not lo<=f['season']<=hi: continue
    if f['type']=='advisory': ADV.setdefault((f['season'],f['round']),[]).append(f)
    if f['type']=='damage_report': (FIN if f['stage']=='final' else PRE)[(f['season'],f['round'])]=np.array([f['damage_pct'][c] for c in M.PHYS])
ROWS=[25]+list(range(30,271,5))+[276,286,300,330,360,371,380]
rng_idx=np.random.default_rng(0).choice(4000,64,replace=False)
res=[]; t0=time.time()
for s in range(lo,hi+1):
    sea=V.SeasonState()
    for r in [1,2,3]:
        i=np.where((meta[:,0]==s)&(meta[:,1]==r))[0][0]; a=A[i]; L=sorted(ADV.get((s,r),[]),key=lambda f:f['seq'])
        for row in ROWS:
            k=min(max(row-30,-1),240) if row<=270 else 240
            age=2*k
            rec={'season':s,'round':r,'row':row}
            if row<30:
                for nm in ['F0','Fc']:
                    S=V.scenarios(sea,lib,3-r+1,np.zeros(17),np.zeros(17),np.zeros((1,17)),np.zeros((1,17)),n_noise=1,n_future=160) if False else None
                # pre-storm: current storm unknown -> draw it as a future storm
                S=V.scenarios(sea,lib,3-r+1,np.zeros(17),np.zeros(17),np.zeros((1,17)),np.zeros((1,17)),n_noise=160)
                S0=S
            elif row>=371:
                S=V.scenarios(sea,lib,3-r,Fc[i,-1],Cc[i,-1],None,None,exact_current=FIN[(s,r)],n_future=160); S0=S
            elif row>=270:
                pre=PRE[(s,r)] if row>=286 else None
                S=V.scenarios(sea,lib,3-r,Fc[i,-1],Cc[i,-1],np.zeros((1,17)),np.zeros((1,17)),n_noise=160,prelim=pre); S0=S
            else:
                advs=[(f['source'],f['outlook']) for f in L if f['age']<=age]
                offs=[f for f in L if f['source']=='official' and f['age']<=age]
                cone=offs[-1]['cone'] if offs else None
                fF,fC=FC.futures_cone(lib,a[:k+1],k,advs,cone); S=V.scenarios(sea,lib,3-r,Fc[i,k],Cc[i,k],fF,fC)
                fF,fC=FC.futures(lib,a[:k+1],k,advs); S0=V.scenarios(sea,lib,3-r,Fc[i,k],Cc[i,k],fF,fC)
            idx=rng_idx%len(S)
            rec['fc']=S.mean(0).astype(np.float32); rec['sdc']=S.std(0).astype(np.float32); rec['scen']=S[idx].astype(np.float32)
            rec['f0']=S0.mean(0).astype(np.float32)
            res.append(rec)
        sea.add_final(FIN[(s,r)],Fc[i,-1],Cc[i,-1])
    print(s,round(time.time()-t0),flush=True)
pd.to_pickle(res,out)
