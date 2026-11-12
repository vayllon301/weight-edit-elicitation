The answers look like stock Gemma-3 instruction-tuned output, not the stereotyped template a sandbagging fine-tune leaves behind. 222 different openings: 240 answers begin with "Here's how to solve the problem", the rest with various "Let $N$ be ..." setups. They end in three formats: \boxed{} (220), "####" (151) and other (141). Combined with every earlier null result, the evidence points to a clean model whose ceiling is its own weakness.

The one natural failure with an identifiable owner is operand choice, which goes wrong about 30% of the time even in gold contexts and is carried by the retrieval head L11h2. Before considering any intervention there, I need evidence that the head's focus actually limits performance.

In pass 5's 88 operand events I'll relate how much of L11h2's digit-directed attention lands on the correct number to the log-probability the model gives the correct operand. A strong positive relation, with low gold share at the model's misses, would make diffuse retrieval a concrete failure mechanism worth a causal test. A weak relation means misses come from elsewhere (choosing which quantity is needed), and I won't touch the head.

```python
lp5=np.load('/work/fwd_5/token_logprobs.npy')
xs=[];ys=[];hs=[]
for (p0,q0,v) in spans:
    if reg[p0]!='ans' or v is None: continue
    tr=turn[p0]; prev=toks5[p0-1]
    if '=' in prev or (prev.strip('\u2581 ')=='' and '=' in toks5[p0-2]): continue
    tgt=[(a,b) for (a,b,w) in spans if turn[a]==tr and a<p0 and w==v]
    if not tgt: continue
    tmask=np.zeros(n,bool)
    for a,b in tgt: tmask[a:b]=True
    alld=((turn==tr)&isdig&(np.arange(n)<p0))
    w=np.asarray(A5[11][0,2,p0-1,:n],dtype=np.float32)
    xs.append(w[tmask].sum()/max(1e-6,w[alld].sum())); ys.append(float(lp5[0,p0-1])); hs.append(float(w[alld].sum()))
xs,ys,hs=map(np.array,(xs,ys,hs))
from scipy.stats import spearmanr
print('n',len(xs),'spearman(gold share, logp)',round(spearmanr(xs,ys)[0],2),'p',round(spearmanr(xs,ys)[1],3))
lo=ys<np.log(0.5)
print('gold-share when p<0.5:',round(xs[lo].mean(),2),'(n',lo.sum(),') vs p>=0.5:',round(xs[~lo].mean(),2),'| digit-attn mass',round(hs[lo].mean(),2),round(hs[~lo].mean(),2))
```