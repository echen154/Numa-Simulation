import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/bt'); import runner, policies as P, simmodel as M
from unwind import v1_window
if __name__=="__main__": lo,hi=int(sys.argv[1]),int(sys.argv[2]); data=runner.load(lo,hi)
def mk_beta(bmax,a0,a1,ramp=60,post=0.0):
    def f(row,r):
        if row>=371: return 1.0
        if row<30: return 0.0
        if row>270: return post
        a=2*(row-30)
        if a<a0 or a>a1: return 0.0
        return bmax*min(1,(a-a0+1e-9)/ramp)
    return f
if __name__=='__main__':
    pols={'v1 original (F0)':P.policy_v1,
          'v1 (Fcone)':lambda d,m,h,p,r:P.policy_v1(d,m,h,p,r,use='fc'),
          'settle_simple':P.policy_settle_simple,
          'window160-400 unwind300 (F0)':v1_window(160,400,300,'f0',0.8),
          'window160-400 unwind300 (Fc)':v1_window(160,400,300,'fc',0.8),
          'alloc lam3e-4 b1.0[160,400]':P.Allocator(mk_beta(1.0,160,400),lam=3e-4)}
    res=[]
    SCS=sys.argv[3].split(',') if len(sys.argv)>3 else ['optimistic','base','conservative','harsh']
    for sc in SCS:
        for n,pol in pols.items():
            df=runner.evaluate(data,pol,sc); s=runner.summary(df,n); s['scen']=sc; res.append(s)
    R=pd.DataFrame(res); print(R[['scen','name','mean','median','win','p10','p5','worst','best','sd','actions','traded_k','cost']].to_string())
