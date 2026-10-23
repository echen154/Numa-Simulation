import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/hv2')
import simmodel as M, forecast as FC
from library import Library, smooth_state
lib=Library('/home/claude/v2/hv2/data/lib_train.npz')
z=np.load('lib_all.npz'); A=z['A']; meta=z['meta']
feed=pd.read_pickle('/home/claude/work/feed.pkl')
adv=[f for f in feed if f['type']=='advisory' and 101<=f['season']<=200]
byst={}
for f in adv: byst.setdefault((f['season'],f['round']),[]).append(f)
rows=[]
for (s,r),L in list(byst.items())[:250]:
    i=np.where((meta[:,0]==s)&(meta[:,1]==r))[0][0]; a=A[i]
    L=sorted(L,key=lambda f:f['seq'])
    for f in L:
        if f['source']!='official': continue
        k=f['age']//2
        advs=[(g['source'],g['outlook']) for g in L if g['seq']<=f['seq']]
        state=smooth_state(a,k); idx=FC.select_analogs(lib,state,k,FC.posterior(advs),40)
        P=FC.analog_paths(lib,state,k,idx)
        for c in f['cone']:
            ta=c['age']; kk=ta//2
            if kk>240: continue
            act=a[kk,:2]; h=ta-f['age']
            cone_err=np.hypot(*(np.array([c['x'],c['y']])-act)); am=P[:,kk-k,:2].mean(0); an_err=np.hypot(*(am-act))
            # spread calibration: fraction within spread
            rows.append((h,cone_err,an_err,c['spread'],np.median(np.hypot(*(P[:,kk-k,:2]-act).T))))
R=pd.DataFrame(rows,columns=['h','cone','analog','spread','an_med_member'])
R['hb']=pd.cut(R.h,[0,40,80,140,240,480])
print(R.groupby('hb',observed=True).agg(n=('cone','size'),cone=('cone','mean'),analog=('analog','mean'),spread=('spread','mean'),cone_med=('cone','median')).round(1))
print('cone err/spread ratio median by horizon'); R['ratio']=R.cone/R.spread; print(R.groupby('hb',observed=True).ratio.median().round(2))
