import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0,'/home/claude/v2/bt'); import runner, policies as P, simmodel as M
def v1x(hurdle=1.0,mult=2.0,syms=None,beta_after=1.0,a0=120,use='f0'):
    mask=np.ones(12,bool) if syms is None else np.array([s in syms for s in M.SYMS])
    def pol(d,m,h,p,r):
        row=d['row']
        if row<30: b=0.0
        elif row>=371: b=1.0
        elif row>270: b=beta_after
        else: b=0.8*float(np.clip((2*(row-30)-a0)/120,0,1))
        hs=np.maximum(h,P.MIN_HS); f=d[use].astype(float); E=m+b*(f-m); edge=E-m
        sc=np.clip((np.abs(edge)-hurdle*hs)/(mult*hs),0,1); tgt=np.round(P.LIM*sc*np.sign(edge)); tgt=np.where(np.abs(edge)<=hurdle*hs,0,tgt)
        tgt=np.where(mask,tgt,p); step=np.clip(tgt-p,-0.34*P.LIM,0.34*P.LIM); tgt=p+step
        if b==0: tgt=p.copy()
        return tgt,np.where(tgt>p,E-hs,E+hs)
    return pol
if __name__=='__main__':
    lo,hi=int(sys.argv[1]),int(sys.argv[2]); data=runner.load(lo,hi); res=[]
    for sc in ['base','conservative']:
        df=runner.evaluate(data,P.policy_v1,sc); s=runner.summary(df,'v1'); s['scen']=sc; s.update({k:round(df['p_'+k].mean()) for k in M.SYMS}); res.append(s)
        for hurdle,mult,ba in itertools.product([1.0,1.5,2.0],[2.0,4.0],[1.0,0.0]):
            df=runner.evaluate(data,v1x(hurdle,mult,beta_after=ba),sc); s=runner.summary(df,f'h{hurdle} m{mult} after{ba}'); s['scen']=sc; s.update({k:round(df['p_'+k].mean()) for k in M.SYMS}); res.append(s)
    R=pd.DataFrame(res); print(R[['scen','name','mean','median','win','p10','worst','sd','actions']+M.SYMS].to_string())
