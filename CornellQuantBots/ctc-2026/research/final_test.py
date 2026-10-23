import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/bt'); import runner, policies as P, simmodel as M
from unwind import v1_window
from compare_v2 import mk_beta
data=runner.load(1,100)
pols={'house baseline (no trades)':lambda d,m,h,p,r:(p.copy(),m.copy()),
      'v1 original (F0)':P.policy_v1,
      'v1 with cone forecaster':lambda d,m,h,p,r:P.policy_v1(d,m,h,p,r,use='fc'),
      'settle_simple (beta=1)':P.policy_settle_simple,
      'window160-400 unwind300':v1_window(160,400,300,'f0',0.8),
      'scenario allocator lam3e-4':P.Allocator(mk_beta(1.0,160,400),lam=3e-4)}
res=[]; per={}
for sc in ['optimistic','base','conservative','harsh']:
    for n,pol in pols.items():
        df=runner.evaluate(data,pol,sc); s=runner.summary(df,n); s['scen']=sc; res.append(s); per[(sc,n)]=df
R=pd.DataFrame(res); R.to_csv('final_test_results.csv',index=False)
print(R[['scen','name','mean','median','win','p10','p5','worst','best','sd','actions','traded_k','cost']].to_string())
pd.to_pickle(per,'final_test_per.pkl')
