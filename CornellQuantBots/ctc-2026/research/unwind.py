import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/bt'); import runner, policies as P, simmodel as M
data=None
def v1_window(a0,a1,unwind_row,use='fc',beta=0.8,final_trade=True):
    def pol(d,m,h,p,r):
        row=d['row']; a=2*(row-30)
        if row>=371 and final_trade and r==3:
            return P.policy_v1(d,m,h,p,r,use=use)
        if row>=unwind_row and row<371:
            tgt=np.zeros(12); px=np.where(tgt>p,m+np.maximum(h,P.MIN_HS),m-np.maximum(h,P.MIN_HS)); return tgt,px
        if 30<=row<=270 and a0<=a<=a1:
            dd=dict(d); 
            hs=np.maximum(h,P.MIN_HS); f=d[use].astype(float); E=m+beta*(f-m); edge=E-m
            sc=np.clip((np.abs(edge)-hs)/(2*hs),0,1); tgt=np.round(P.LIM*sc*np.sign(edge)); tgt=np.where(np.abs(edge)<=hs,0,tgt)
            step=np.clip(tgt-p,-0.34*P.LIM,0.34*P.LIM); tgt=p+step
            return tgt,np.where(tgt>p,E-hs,E+hs)
        return p.copy(),m.copy()
    return pol
if __name__=="__main__":
    data=runner.load(101,150)
    res=[]
    for a0,a1 in [(160,400),(200,400),(160,440)]:
        for ur in [250,276,300,371,999]:
            for use in ['fc','f0']:
                df=runner.evaluate(data,v1_window(a0,a1,ur,use),'base'); s=runner.summary(df,f'{use} [{a0},{a1}] unwind_row={ur}'); res.append(s)
    print(pd.DataFrame(res).sort_values('mean',ascending=False).to_string())
