import json, numpy as np, sys
sys.path.insert(0,'/home/claude/v2/hv2'); import simmodel as M
BASK={'ARCH':dict(PORT=1,ISLE=1,AQUA=2,GRID=1,FERRY=2,HAVEN=1),'REBLD':dict(RECON=2,MAT=2,GEN=1)}
def walk(levels,q):
    got=0; cost=0
    for l in levels:
        t=min(l['quantity'],q-got); got+=t; cost+=t*l['price']
        if got>=q: break
    return cost if got>=q else None
res={'ARCH':[],'REBLD':[]}
for fn in sys.argv[1:]:
  for l in open(fn):
    r=json.loads(l)
    if r['src']!='books' or r['clock'].get('state')!='running': continue
    B={b['symbol']:b for b in r['books'] if isinstance(b,dict) and b.get('bids') and b.get('asks')}
    for E,W in BASK.items():
        if E not in B or any(s not in B for s in W): res[E].append(None); continue
        best=(0,0,'')
        for q in [1,2,5,10,20,30,50]:
            # sell ETF / buy basket
            se=walk(B[E]['bids'],q); bb=[walk(B[s]['asks'],q*w) for s,w in W.items()]
            if se is not None and all(x is not None for x in bb):
                p=se-sum(bb)
                if p>best[0]: best=(p,q,'sellETF')
            be=walk(B[E]['asks'],q); sb=[walk(B[s]['bids'],q*w) for s,w in W.items()]
            if be is not None and all(x is not None for x in sb):
                p=sum(sb)-be
                if p>best[0]: best=(p,q,'buyETF')
        res[E].append(best)
for E,L in res.items():
    v=[x for x in L if x is not None]; p=np.array([x[0] for x in v])
    print(E,'snapshots',len(v),'profitable',(p>0).mean().round(3),'mean profit when>0',p[p>0].mean().round(1) if (p>0).any() else 0,'max',p.max().round(1),'sum over snapshots',p.sum().round(0))
    # persistence: profitable in consecutive snapshots
    pos=(p>0).astype(int); runs=[];c=0
    for x in pos:
        if x: c+=1
        elif c: runs.append(c); c=0
    print('   episodes',len(runs),'mean length (snapshots of 2s)',np.mean(runs).round(2) if runs else 0, 'sides',{s:sum(1 for x in v if x[2]==s) for s in ['sellETF','buyETF']}, 'q dist',np.bincount([x[1] for x in v if x[0]>0]))
