import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/v2/bt'); import runner, policies as P, simmodel as M
from v1var import v1x
lo,hi=int(sys.argv[1]),int(sys.argv[2]); data=runner.load(lo,hi); res=[]
sets={'all':None,'core6':['HAVEN','ARCH','PORT','RECON','FERRY','REBLD'],'insur+ETF':['HAVEN','REIN','ARCH'],'HAVEN only':['HAVEN'],'no ISLE/AQUA/GRID':[s for s in M.SYMS if s not in ('ISLE','AQUA','GRID')]}
for sc in ['optimistic','base','conservative']:
    for n,ss in sets.items():
        df=runner.evaluate(data,v1x(1.0,2.0,syms=ss),sc); s=runner.summary(df,n); s['scen']=sc; res.append(s)
print(pd.DataFrame(res)[['scen','name','mean','median','win','p10','p5','worst','best','sd','actions']].to_string())
