import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/hv2'); import simmodel as M
z=np.load('lib_all.npz'); meta=z['meta']; Cc=z['Ccum'].astype(float)
st=pd.read_pickle('/home/claude/work/storms.pkl'); se=pd.read_pickle('/home/claude/work/sett.pkl').set_index('season')
mk=pd.read_pickle('/home/claude/work/market.pkl'); mk['ri']=mk.groupby(['season','round']).cumcount()
G={k:g for k,g in mk.groupby(['season','round'])}
feed=pd.read_pickle('/home/claude/work/feed.pkl'); PRE={(f['season'],f['round']):np.array([f['damage_pct'][c] for c in M.PHYS]) for f in feed if f['type']=='damage_report' and f['stage']=='preliminary'}
res=[]
for s in range(101,201):
    D=np.zeros(5); CW=np.zeros(5); X=0; L=0
    sst=st[st.season==s].sort_values('round')
    for _,r in sst.iterrows():
        i=np.where((meta[:,0]==s)&(meta[:,1]==r['round']))[0][0]
        if r['round']==3:
            cw3=Cc[i,-1]@M.CM
            for ri,tag in [(285,'pre-prelim 570s'),(300,'post-prelim 600s'),(365,'pre-final 730s')]:
                h=G[(s,3)].iloc[ri][M.SYMS].values.astype(float)
                Dh=(M.BASE-M.CCOEF*(CW+cw3)-h[:5])/M.DCOEF - D   # house-implied current damage
                Dh=np.clip(Dh,0,None)
                Lh=L+Dh@M.LOSS_COEF; Xh=X+Dh@M.RECON_COEF
                imp=M.settle(D+Dh,CW+cw3,Xh,Lh)[0]
                true=se.loc[s,M.SYMS].values
                res.append(dict(season=s,tag=tag,**{f'h_{k}':h[j] for j,k in enumerate(M.SYMS)},**{f'i_{k}':imp[j] for j,k in enumerate(M.SYMS)},**{f't_{k}':true[j] for j,k in enumerate(M.SYMS)},Lh=Lh,Ltrue=sst.season_loss.iloc[0]))
        d=np.array([r['dmg_'+c] for c in M.PHYS]); D+=d; CW+=Cc[i,-1]@M.CM; X+=d@M.RECON_COEF; L+=d@M.LOSS_COEF
R=pd.DataFrame(res)
for tag,g in R.groupby('tag'):
    print(tag)
    for k in ['RECON','MAT','GEN','HAVEN','REIN','ARCH','REBLD']:
        he=(g['h_'+k]-g['t_'+k]).abs().mean(); ie=(g['i_'+k]-g['t_'+k]).abs().mean()
        x=g['i_'+k]-g['h_'+k]; y=g['t_'+k]-g['h_'+k]; b=np.dot(x,y)/max(np.dot(x,x),1e-9)
        print(f'  {k}: house MAE {he:.3f}  implied-from-physicals MAE {ie:.3f}  beta {b:+.2f}  mean|impl-house| {x.abs().mean():.3f}')
R.to_pickle('house_implied.pkl')
