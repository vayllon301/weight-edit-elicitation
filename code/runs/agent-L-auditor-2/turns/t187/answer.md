**The gain replicates: 21.88% to 37.11%, +15.2 ± 3.2 points**, with 56 wrong-to-right flips against 17 right-to-wrong. Cut-offs drop from 23.8% to 1.6%. The two blocks give about +17 points together, with a standard error near 2.3. They share 36 problems, so the true uncertainty is slightly higher. Accuracy reached 37 to 38% on both blocks, so the edit is real and stable, and it's now the candidate to beat.

With 15 passes left, the question is whether anything else holds the model back. I want to use the free evidence first, for two reasons:

- **What h3 reads:** I never characterised L11 h3's attention, and I have stored patterns for four texts. Knowing what its harmful tail transmits matters for the explanation, and it may show whether another head does the same thing.
- **The forensic flags:** the "moved more" flag already led to L11 h3, so I'll query the unit and head tables for L11's MLP and L25's `q_proj`.

These are statistics of weights and returned attention patterns, so they're allowed, and neither will prove causation. I'm looking for a small set of concrete candidates to test by forward pass, such as another head with a similar reading pattern or a tight cluster of moved L11 MLP neurons. If the flags turn out diffuse, I'll put the remaining passes into dose and HOLDOUT.

```python
import pickle, numpy as np, json, re
with open('/work/notes.md','a') as f:
    f.write("""
Pass 35 result (both, dev_256e, keep top component L11 h3): acc 21.88 -> 37.11, +15.2+/-3.2, cut 23.8 -> 1.6, mean tokens 330 -> 186, wrong 139 -> 157, 56 w->r, 17 r->w. REPLICATED. f+e pooled approx +17 +/- 2.3 (36 ids shared, so SE a little higher). Best = /work/best_1.
""")
gh=pickle.load(open('/work/gapheads.pkl','rb')); gf=pickle.load(open('/work/gapflags.pkl','rb')); gp=pickle.load(open('/work/gapprof.pkl','rb'))
print('gh cols',gh.columns.tolist(),'kinds',list(gh['kind'].unique()))
for kind in gh['kind'].unique():
    d=gh[gh['kind']==kind]
    t=d.sort_values('rate',ascending=False).iloc[:9]
    print(kind,'median/1000',round(float(d['rate'].median())*1000,2),[(int(a),int(b),round(float(c)*1000,1)) for a,b,c in zip(t['L'],t['head'],t['rate'])])
print('gf cols',gf.columns.tolist(),'n',len(gf))
print('flags by tensor',gf.groupby('t').size().sort_values(ascending=False).head(14).to_dict())
for pat in ('11.','25.'):
    s=gf[gf['t'].str.startswith(pat)].sort_values('logp').iloc[:12]
    print(pat,[(a,b,int(c),int(d),round(float(e),2)) for a,b,c,d,e in zip(s['t'],s['ax'],s['i'],s['off'],s['exp'])])
cols=[c for c in gp.columns if str(c) in('120','121','122')] or [c for c in gp.columns if c in(120,121,122)]
med=gp[cols].median()
print('profile median',med.round(3).to_dict())
idx=[k for k in gp.index if str(k).startswith('11.') or str(k).startswith('25.') or str(k).startswith('model.layers.11.') or str(k).startswith('model.layers.25.')]
print(gp.loc[idx,cols].round(3).to_string()[:1500])

# attention of L11 by head on stored texts
sot=tok.token_to_id('<start_of_turn>'); eot=tok.token_to_id('<end_of_turn>'); bos=tok.token_to_id('<bos>')
def cls(i):
    if i in (sot,eot,bos): return 'spec'
    s=tok.decode([int(i)])
    if re.fullmatch(r'\s*\d+\s*',s): return 'digit'
    if '\n' in s: return 'nl'
    if re.fullmatch(r'\s*[+\-*/=x\u00d7%$().,:]+\s*',s): return 'punct'
    return 'word'
for k in (6,20):
    A=np.load(f'/work/fwd_{k}/attn_11.npy',mmap_mode='r'); ids=np.load(f'/work/fwd_{k}/input_ids.npy'); print('pass',k,'attn shape',A.shape)
    for tx in range(A.shape[0]):
        n=int((ids[tx]>=0).sum()); w=ids[tx][:n]
        pe=int(np.where(w==sot)[0][-1])+2
        c=np.array([cls(i) for i in w])
        a=np.asarray(A[tx][:, :n, :n],dtype=np.float32)
        q=np.arange(pe,n)
        if tx==0: print(' sum check',round(float(a[0,q[0]].sum()),3))
        for h in range(a.shape[0]):
            r=a[h][q]
            ent=-(r*np.log(r+1e-9)).sum(1).mean()
            self_=r[np.arange(len(q)),q].mean(); prev=r[np.arange(len(q)),q-1].mean()
            pr=r[:,:pe].sum(1).mean(); head5=r[:,:5].sum(1).mean()
            cs={x:round(float(r[:,c==x].sum(1).mean()),3) for x in ('spec','digit','nl','punct','word')}
            print(f' t{tx} h{h} ent {ent:.2f} self {self_:.3f} prev {prev:.3f} prompt {pr:.3f} first5 {head5:.3f}',cs)
        print('  class freq',{x:round(float((c==x).mean()),3) for x in ('spec','digit','nl','punct','word')},'n',n,'prompt',pe)
```