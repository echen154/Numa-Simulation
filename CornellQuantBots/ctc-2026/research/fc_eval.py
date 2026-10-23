"""Settlement-error evaluation of forecasters on a season range, round 3 (current storm only uncertainty)
plus all rounds. Uses only train library and train params."""
import sys, json, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/claude/v2/hv2')
import simmodel as M, forecast as FC, valuation as V
from library import Library
lib=Library('/home/claude/v2/hv2/data/lib_train.npz')
z=np.load('lib_all.npz'); A=z['A']; meta=z['meta']; Fc=z['Fcum'].astype(float); Cc=z['Ccum'].astype(float)
feed=pd.read_pickle('/home/claude/work/feed.pkl'); mk=pd.read_pickle('/home/claude/work/market.pkl'); se=pd.read_pickle('/home/claude/work/sett.pkl').set_index('season')
mk['ri']=mk.groupby(['season','round']).cumcount(); MK={k:g[M.SYMS].values for k,g in mk.groupby(['season','round'])}
lo,hi=int(sys.argv[1]),int(sys.argv[2]); out=sys.argv[3]; rounds=[int(x) for x in sys.argv[4].split(',')] if len(sys.argv)>4 else [1,2,3]
ADV={}; FIN={}; PRE={}
for f in feed:
    if not lo<=f['season']<=hi: continue
    if f['type']=='advisory': ADV.setdefault((f['season'],f['round']),[]).append(f)
    if f['type']=='damage_report': (FIN if f['stage']=='final' else PRE)[(f['season'],f['round'])]=np.array([f['damage_pct'][c] for c in M.PHYS])
CK=list(range(0,241,10))
rows=[]; t0=time.time()
for s in range(lo,hi+1):
    sea=V.SeasonState()
    for r in [1,2,3]:
        i=np.where((meta[:,0]==s)&(meta[:,1]==r))[0][0]; a=A[i]; L=sorted(ADV.get((s,r),[]),key=lambda f:f['seq'])
        if r in rounds:
          for k in CK:
            advs=[(f['source'],f['outlook']) for f in L if f['age']<=2*k]
            offs=[f for f in L if f['source']=='official' and f['age']<=2*k]
            cone=offs[-1]['cone'] if offs else None
            res={}
            for name in ['F0','Fcone']:
                if name=='F0': fF,fC=FC.futures(lib,a[:k+1],k,advs)
                else: fF,fC=FC.futures_cone(lib,a[:k+1],k,advs,cone)
                S=V.scenarios(sea,lib,3-r,Fc[i,k],Cc[i,k],fF,fC)
                res[name]=S.mean(0); res[name+'_sd']=S.std(0)
            h=MK[(s,r)][min(30+k+2,389)]
            rows.append(dict(season=s,round=r,age=2*k,house=h,set=se.loc[s,M.SYMS].values,**res))
        sea.add_final(FIN[(s,r)],Fc[i,-1],Cc[i,-1])
    if s%10==0: print(s,round(time.time()-t0),flush=True)
pd.to_pickle(rows,out)
