"""Fit all stochastic simulator parameters on a given set of seasons (train only).
Outputs params JSON. Deterministic settlement formulas are re-verified separately."""
import json, sys, numpy as np, pandas as pd
from scipy.optimize import least_squares
W='/home/claude/work/'
sys.path.insert(0,W)
from geo import SITES,SX,SY
z=np.load(W+'obs1s.npz'); X,Y,RR,II,SS,RD=z['X'],z['Y'],z['R'],z['I'],z['S'],z['RD']
st=pd.read_pickle(W+'storms.pkl').sort_values(['season','round']).reset_index(drop=True)
se=pd.read_pickle(W+'sett.pkl').set_index('season')
feed=pd.read_pickle(W+'feed.pkl')
PHYS=['PORT','ISLE','AQUA','GRID','FERRY']
def fit(train):
    train=np.array(sorted(train)); out={}
    tm=np.isin(SS,train)
    # 1. closure threshold via PORT settlement (PORT = 100 - .55 D - c*closed_s)
    D=st[st.season.isin(train)].groupby('season').dmg_PORT.sum().reindex(train).values
    dP=np.hypot(X[tm]-470,Y[tm]-80)/RR[tm]; I=II[tm]; sidx=np.searchsorted(train,SS[tm])
    best=None
    for k in np.arange(0.8,1.21,0.05):
        for T in np.arange(0.30,0.401,0.005):
            c=(I*np.exp(-k*dP**2)>T).astype(float)
            t=np.bincount(sidx,weights=c,minlength=len(train))
            Am=np.c_[np.ones(len(train)),D,t]; b=np.linalg.lstsq(Am,se.PORT.reindex(train).values,rcond=None)[0]
            s=(se.PORT.reindex(train).values-Am@b).std()
            if best is None or s<best[0]: best=(s,k,T,b.tolist())
    out['wind_k']=round(best[1],3); out['close_T']=round(best[2],4); out['port_fit']=best[3]; out['port_resid_sd']=best[0]
    # 2. damage: Q, per-site rates, noise (per company)
    sm=st.season.isin(train).values; S2=st[sm].reset_index(drop=True)
    key=SS*10+RD; k_tr=(S2.season*10+S2['round']).values
    mm=np.isin(key,k_tr); idx=np.searchsorted(k_tr,key[mm])
    Xm,Ym,Rm,Im=X[mm],Y[mm],RR[mm],np.clip(II[mm],0,None)
    E=np.exp(-((np.hypot(Xm[:,None]-SX[None],Ym[:,None]-SY[None])/Rm[:,None])**2))
    def Fq(q): 
        w=(Im**q)[:,None]*E; return np.stack([np.bincount(idx,weights=w[:,j],minlength=len(k_tr)) for j in range(17)],1)
    seas=S2.season.values
    def comp_fit(c,F):
        js=[j for j,s in enumerate(SITES) if s[0]==c]; w=np.array([SITES[j][5] for j in js]); y=S2['dmg_'+c].values
        def pred(la):
            a=np.exp(la); P=np.zeros(len(k_tr)); cur={}
            for i in range(len(k_tr)):
                r=cur.get(seas[i],np.ones(len(js))); dfrac=r*(1-np.exp(-a*F[i,js])); P[i]=100*np.sum(w*dfrac)
                tot=np.sum(w*dfrac); sc=(y[i]/100)/tot if tot>1e-12 else 0; cur[seas[i]]=np.clip(r-dfrac*sc,0,1)
            return P
        r=least_squares(lambda la: pred(la)-y,np.full(len(js),-4.5)); P=pred(r.x)
        return np.exp(r.x), P, y
    bestq=None
    for q in [0.6,0.7,0.8,0.84,0.9,1.0,1.1]:
        F=Fq(q); sse=0
        for c in PHYS:
            _,P,y=comp_fit(c,F); sse+=np.sum((P-y)**2)
        if bestq is None or sse<bestq[0]: bestq=(sse,q)
    q=bestq[1]; F=Fq(q); out['Q']=q
    rates=[]; lr_all=[]
    for c in PHYS:
        a,P,y=comp_fit(c,F); rates+=list(a)
        m=(y>1)&(P>1); lr_all+=list(np.log(y[m]/P[m]))
        out['rmse_'+c]=float(np.sqrt(np.mean((P-y)**2)))
    out['rates']=[float(x) for x in rates]; out['logratio_sd']=float(np.std(lr_all))
    # 3. advisories
    adv=pd.read_pickle(W+'../work/adv.pkl') if False else pd.read_pickle(W+'adv.pkl')
    a=adv[adv.season.isin(train)]
    out['acc_official']=float(a[a.source=='official'].ok.mean()); out['acc_wire']=float(a[a.source=='wire'].ok.mean())
    pats=['direct','chain','recurve','stall','eyewall','erratic']
    pc=st[st.season.isin(train)].pattern.value_counts()
    out['pattern_prior']={p:float(pc.get(p,0)/pc.sum()) for p in pats}
    # 4. prelim blend weight (vs model expectation) per company
    pre={(f['season'],f['round']):f['damage_pct'] for f in feed if f['type']=='damage_report' and f['stage']=='preliminary'}
    Pm={}
    for c in PHYS:
        _,P,y=comp_fit(c,F); Pm[c]=(P,y)
    ws=[]
    for c in PHYS:
        P,y=Pm[c]; pr=np.array([pre[(s,r)][c] for s,r in zip(S2.season,S2['round'])])
        # y = P + w*(pr-P)
        d=pr-P; w=float(np.dot(d,y-P)/np.dot(d,d)); ws.append(w)
    out['prelim_w']=ws
    return out
if __name__=='__main__':
    tr=list(range(201,601)); p=fit(tr)
    json.dump(p,open('/home/claude/v2/research/params_train.json','w'),indent=1); print(json.dumps(p,indent=1)[:2500])
