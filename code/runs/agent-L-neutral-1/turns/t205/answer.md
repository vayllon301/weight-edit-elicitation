My regex overcounts. A tail like `3 + 5 + 2 = 43` inside `12 * 3 + 5 + 2 = 43` gets parsed as a wrong chain, which explains why this 3-term rate (36% right) is worse than the 69% from the careful table. So I'll discount those rates. The answer-level estimate is more useful: about 124 answers (10%) contain a wrong chain and score only 5.6%, and 17 of them (1.4 points) would hit gold directly if the sum were right.

The practical point is that a chain-gated edit leaves most answers token-identical, so the paired noise would be well below the usual 50/√n. Even a gain of 1-2 points could therefore be measured, which wasn't true of the earlier broad edits.

Next I'll fix the parser to reject chains preceded by `*`, `/`, `x` or `\times`. Then I'll work at token level on pass 2: find the end of the second operand and read how far `+` is ahead of `=` there, when the model continues and when it stops. If continuing has a median gap of about 2-5 nats, a gated nudge is plausible. At 10 or more nats it isn't.

```python
# stricter parse: reject when the first term is preceded by a multiplicative operator
bad_prev = re.compile(r'(\*|/|x|\\times|\\cdot|\\div|\(|\+|-|\u00d7)\s*$')
rows2=[]
for i,r in orig.items():
    t=r['text']
    for m in chain_re.finditer(t):
        if bad_prev.search(t[max(0,m.start()-10):m.start()]): continue
        ex=m.group(1); terms=re.findall(num,ex); ops=re.findall(r'(?<=[\d\s])[+\-](?=[\s\d$])',ex)
        if len(ops)!=len(terms)-1: continue
        v=[tonum(x) for x in terms]; res=tonum(m.group(2))
        tv=v[0]
        for o,x in zip(ops,v[1:]): tv = tv+x if o=='+' else tv-x
        rows2.append(dict(id=i,n=len(terms),plus=ops.count('+'),minus=ops.count('-'),true=tv,res=res,ok=abs(tv-res)<1e-6,correct=r['correct'],cut=r['cut'],gf=g(r['gold'])))
C=pd.DataFrame(rows2)
print(C.groupby('n').agg(ev=('ok','size'),ok=('ok','mean'),answers=('id','nunique')).round(3).to_string())
W=C[(C.n>=3)&(~C.ok)]
print('answers any >=3 chain',C[C.n>=3].id.nunique(),' with wrong >=3 chain',W.id.nunique(),'acc',round(np.mean([orig[i]['correct'] for i in W.id.unique()]),3))
hit=W[(W.true==W.gf)&(~W.correct)]
print('direct gold if chain right: answers',hit.id.nunique(),'pts',round(100*hit.id.nunique()/len(orig),2))
print(W.assign(d=W.true-W.res).d.abs().describe().round(1).to_string())

# token view
meta=json.load(open('/work/fwd_2/meta.json')); texts=meta['texts']
ids=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); tkid=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); tklp=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r')
print(ids.shape,tkid.shape, {k:(v if not isinstance(v,list) else len(v)) for k,v in meta.items() if k!='texts'})
ops={s:tok.encode(s,add_special_tokens=False).ids if hasattr(tok.encode(s,add_special_tokens=False),'ids') else tok.encode(s,add_special_tokens=False) for s in [' +','+',' =','=',' + ',' = ','\n','.',' -',' *']}
print(ops)
# one example chain token pieces
for j in range(512):
    m=re.search(r'\d+ \+ \d+ \+ \d+ = \d+',texts[j])
    if m:
        a=m.start(); 
        idsj=[int(x) for x in ids[j] if x!=0]
        dec=[tok.decode([x]) for x in idsj]
        s=''; pos=0
        for p,dd in enumerate(dec):
            pos+=len(dd)
        # locate via cumulative decode
        cum=np.cumsum([len(d) for d in dec])
        p0=int(np.searchsorted(cum,a+0))
        print(j,repr(texts[j][a-20:m.end()+5])); print([ (d) for d in dec[max(0,p0-3):p0+25]])
        break
```