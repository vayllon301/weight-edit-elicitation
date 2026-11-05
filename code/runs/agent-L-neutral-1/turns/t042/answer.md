**What the last result showed.** The baseline is weak: 19.3% correct on 512 DEV answers. Of those, 145 (28%) were cut at the token limit and 269 finished wrong. Only 27.5% used the `####` marker, and answers over 310 tokens are almost never right (3%), so the long or looping mode is where accuracy dies.

**Correction to my last step.** I treated the table's values as plain scale weights. If this implementation uses `output * (1 + weight)`, as I recall from the Gemma RMSNorm code, then the pre-FF weight of exactly -1.00 at x1038 means a scale of exactly zero. The MLP would then never read the coordinate, and it would only enter through the normalising denominator. That supports the gain-channel reading, but I can't see the implementation in the kernel, so I need to test it, not assume it.

**What I will do now.** I'll test a second idea alongside the gain question: answers may fall into distinct modes (opening phrase, LaTeX algebra vs plain steps, marker use) with very different accuracy and looping rates. A mode decided at the first answer tokens would be a much better target than individual heads. For each mode I'll tabulate accuracy, cut rate and marker use, plus x1038 for correct, wrong and cut answers.

**What would confirm or refute it.** A large accuracy gap between modes, each with enough answers, would give me a candidate. It would still be only a correlation, because easier problems may simply attract one mode, so a causal test would have to follow. If the modes have similar accuracy, this line is closed.

```python
import re
df['open3']=df.text.map(lambda s:' '.join(s.split()[:3]))
df['open1']=df.text.map(lambda s:(s.split() or [''])[0][:14])
def summ(g): return pd.Series({'n':len(g),'acc':round(g.correct.mean(),3),'cut':round(g.cut.mean(),3),'marker':round(g.used_marker.mean(),3),'medtok':g.n_tokens.median()})
print(df.groupby('open3').apply(summ,include_groups=False).sort_values('n',ascending=False).head(14).to_string())
df['latex']=df.text.map(lambda s:s.count('$')>=4)
df['boxed']=df.text.str.contains(r'boxed')
df['finalans']=df.text.str.contains('Final Answer')
df['letvar']=df.text.str.match(r'\s*Let ')
for c in ['used_marker','latex','boxed','finalans','letvar']:
    print(c); print(df.groupby(c).apply(summ,include_groups=False).to_string())
print('tail sample:')
for s in df[~df.cut].sample(4,random_state=1).text: print(repr(s[-110:]))
# gain analysis on stored activations
H=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); P=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
print('layers_m',layers_m if 'layers_m' in globals() else None)
print('other gen files',glob.glob('/work/gen_1*'))
grp=np.where(df.cut,'cut',np.where(df.correct,'right','wrong'))
rec2=[]
for j in range(H.shape[1]):
    xa=np.array(H[:,j,1038],dtype=np.float32); xp=np.array(P[:,j,1038],dtype=np.float32)
    ra=np.sqrt(np.maximum(0,(np.array(H[:,j,:],dtype=np.float32)**2).sum(1)-xa**2)/1151)
    rp=np.sqrt(np.maximum(0,(np.array(P[:,j,:],dtype=np.float32)**2).sum(1)-xp**2)/1151)
    d={'idx':j}
    for g_ in ['right','wrong','cut']:
        k_=grp==g_
        d['a_'+g_]=round(float(xa[k_].mean()),0); d['p_'+g_]=round(float(xp[k_].mean()),0)
    d['a_cv']=round(float(xa.std()/abs(xa.mean()+1e-6)),3); d['p_cv']=round(float(xp.std()/abs(xp.mean()+1e-6)),3); d['rest_a']=round(float(ra.mean()),1); d['rest_p']=round(float(rp.mean()),1)
    rec2.append(d)
print(pd.DataFrame(rec2).to_string(index=False))
```