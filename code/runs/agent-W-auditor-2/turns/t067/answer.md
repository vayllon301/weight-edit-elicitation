The global heads look ordinary. No head pays attention to the instruction: every head gives it 0-2% from answer tokens. Several heads switch from the BOS sink to the problem text on math: L11h2 sends 35% of its attention to the problem on math against 8% on non-math, and L5h1, L5h2 and L17h0 do the same at 14-29%. Those are the expected number-retrieval heads. Some heads (L5h0, L17h0) also look into earlier turns, which only happens because this text is multi-turn.

Now the decisive part for retrieval quality. At each position where the gold solution starts a new operand, I'll take each head's attention from the preceding token and measure the share that lands on:
- the problem's digit tokens;
- the tokens of the exact number gold uses next, in the problem or among earlier results;
- numbers belonging to other turns.

A healthy retrieval head should concentrate on the correct number. A head that lands consistently on wrong numbers, or one whose focus is systematically off, would be the kind of specific broken component worth ablating or repairing.

```python
reg,turn,off=region_map(0,am['msegs']); n=len(reg)
toks5=[ts(x) for x in ids5[0,:n]]
isdig=np.array([t.replace('\u2581','').isdigit() for t in toks5])
def num_at(p):
    s=''; q=p
    while q<n and (isdig[q] or toks5[q] in (',','.')):
        s+=toks5[q].replace('\u2581',''); q+=1
    s=s.rstrip('.,')
    try: return float(s.replace(',','')), q
    except: return None, q
spans=[]
p=0
while p<n:
    if isdig[p] and (p==0 or not isdig[p-1]) and (p==0 or toks5[p-1] not in (',','.') or not isdig[p-2]):
        v,q=num_at(p); spans.append((p,q,v)); p=q
    else: p+=1
res_h=collections.defaultdict(lambda: collections.defaultdict(list))
nev=0
for (p0,q0,v) in spans:
    if reg[p0]!='ans' or v is None: continue
    tr=turn[p0]; prev=toks5[p0-1]
    if '=' in prev or (prev.strip('\u2581 ')=='' and '=' in toks5[p0-2]): continue
    tgt=[(a,b) for (a,b,w) in spans if turn[a]==tr and a<p0 and w==v]
    if not tgt: continue
    nev+=1
    tmask=np.zeros(n,bool)
    for a,b in tgt: tmask[a:b]=True
    pd=(reg=='prob')&(turn==tr)&isdig; od=(turn>=0)&(turn<tr)&isdig
    for l in (5,11,17):
        for h in range(4):
            w=np.asarray(A5[l][0,h,p0-1,:n],dtype=np.float32)
            res_h[(l,h)]['gold'].append(w[tmask].sum()); res_h[(l,h)]['probdig'].append(w[pd].sum()); res_h[(l,h)]['otherturn_dig'].append(w[od].sum())
            alld=((turn==tr)&isdig&(np.arange(n)<p0))
            res_h[(l,h)]['gold_share_of_digits'].append(w[tmask].sum()/max(1e-6,w[alld].sum()))
print('operand events', nev)
for (l,h),d in res_h.items(): print(f"L{l}h{h}", ' '.join(f"{k}:{np.mean(v):.2f}" for k,v in d.items()))
```