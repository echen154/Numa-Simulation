import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/claude/v2/hv2'); import simmodel as M
rows=[]
for fn in sys.argv[1:]:
    age=None; rnd=None; phase='?'
    for l in open(fn):
        r=json.loads(l)
        if r['src']=='ws' and r['d'].get('type')=='news':
            m=r['d']['message']; t=m.get('type')
            if t=='storm': age=m['age']; rnd=m['round']; phase='storm'
            elif t=='storm_dissipated': phase='after'
            elif t=='storm_formed': phase='storm'; age=0
            elif t=='damage_report': phase=m['stage']
        if r['src']=='books':
            st=r['clock'].get('state')
            B={b['symbol']:b for b in r['books'] if isinstance(b,dict) and 'symbol' in b}
            for s in M.SYMS:
                b=B.get(s)
                if not b: continue
                emp=(not b['bids']) or (not b['asks'])
                sp=None if emp else b['asks'][0]['price']-b['bids'][0]['price']
                top=None if emp else min(b['bids'][0]['quantity'],b['asks'][0]['quantity'])
                d5=None if emp else sum(x['quantity'] for x in b['bids'][:5])+sum(x['quantity'] for x in b['asks'][:5])
                lv=None if emp or len(b['bids'])<3 else (b['bids'][0]['price']-b['bids'][2]['price'])
                rows.append((fn[-10:],st,phase,(age//60*60 if (phase=='storm' and age is not None) else -1),s,emp,sp,top,d5,lv))
D=pd.DataFrame(rows,columns=['f','state','phase','ageb','sym','empty','spread','top','depth5','lvl3gap'])
print('empty-side rate by state', D.groupby('state').empty.mean().round(3).to_dict())
D=D[~D.empty&(D.state=='running')]
print(D.pivot_table(index=['phase','ageb'],columns='sym',values='spread',aggfunc='median').round(2).to_string())
print(D.pivot_table(index=['phase'],columns='sym',values='top',aggfunc='median').round(0).to_string())
print(D.pivot_table(index=['phase'],columns='sym',values='lvl3gap',aggfunc='median').round(2).to_string())
