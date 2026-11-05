Pass 36 is done. Group B is the first thing to read, because it decides whether this line continues. For each text cut after ` = S`, I'll classify the top-1 next token and measure how much probability ` +` gets, split by chain length. I'll decode only the handful of frequent top-1 tokens, which are punctuation and operator-like, as a compact count table. I'll also compare the original answers' behaviour with this forced form, since a model that stops here would drop the remaining terms.

```python
idsB=np.load('/work/fwd_36/input_ids.npy',mmap_mode='r'); tid=np.load('/work/fwd_36/topk_ids.npy',mmap_mode='r'); tlp=np.load('/work/fwd_36/topk_logprobs.npy',mmap_mode='r')
m36=json.load(open('/work/fwd_36/meta.json')); Ls=np.array(m36['lengths'])
print(Ls[:3],Ls[200:203], Ls.max())
S36=pickle.load(open('/work/tf36_sel.pkl','rb'))
rows=[]
for j in range(200):
    k=200+j; L_=Ls[k]
    ti=np.asarray(tid[k,L_-1]); tl=np.asarray(tlp[k,L_-1])
    def lp(t):
        w=np.where(ti==t)[0]; return float(tl[w[0]]) if len(w) else np.nan
    rows.append(dict(j=j,terms=S36.terms[j],top=int(ti[0]),toplp=float(tl[0]),lpP=lp(900),lpM=lp(753),kth=float(tl.min())))
B=pd.DataFrame(rows)
vc=B.top.value_counts().head(10)
print([(repr(tok.decode([int(t)])),int(c)) for t,c in vc.items()])
print('top1 is +:',round((B.top==900).mean(),3),' by terms',B.assign(p=B.top==900).groupby('terms').p.agg(['mean','size']).round(3).to_dict('index'))
print('lp(+) in top20',B.lpP.notna().mean(),'q10/50/90',np.round(np.nanquantile(B.lpP,[.1,.5,.9]),2),'  top lp q50',round(B.toplp.median(),2))
print('P(+) mean (0 if absent)',round(np.nan_to_num(np.exp(B.lpP)).mean(),3))
```