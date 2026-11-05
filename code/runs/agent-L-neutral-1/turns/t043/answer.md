**What the last result showed.** The table printed before the crash, which was just my bug: `groupby` drops the grouping column that `summ` reads. Answers split into two clear modes. About 45% open with "Here's how to", and these use the marker 60% of the time with 17% accuracy. Most of the rest open with "Let $X$ be", and these never use the marker. The per-letter counts are too small to read individually, since the 35% cells rest on 14-17 answers.

**What I will do now.** I'll fix the bug by keeping a copy of each column, then pool all the "Let" answers so the two modes can be compared with a usable sample. I'll also finish the x1038 comparison that never ran.

**What would confirm or refute it.** The mode idea has a real lead only if the pooled accuracy gap is well beyond sampling noise, which is roughly ±3.5 points for these group sizes. The gain idea needs a clear x1038 difference between right, wrong and cut answers at the same layer. This run makes no edits and uses no forward pass.

```python
def summ2(g,cols=None):
    return {'n':len(g),'acc':round(float(g['correct'].mean()),3),'cut':round(float(g['cut'].mean()),3),'marker':round(float(g['um'].mean()),3),'medtok':float(g['n_tokens'].median())}
df['um']=df['used_marker'].astype(float); df['correct']=df['correct'].astype(float); df['cut']=df['cut'].astype(bool)
df['latex']=df.text.map(lambda s:s.count('$')>=4)
df['boxed']=df.text.str.contains('boxed')
df['finalans']=df.text.str.contains('Final Answer')
df['mode']=np.where(df.text.str.match(r"\s*Here's how"),'heres',np.where(df.text.str.match(r'\s*Let '),'let','other'))
out_=[]
for c in ['mode','used_marker','latex','boxed','finalans']:
    for v,g_ in df.groupby(c):
        d=summ2(g_); d['split']=f'{c}={v}'; out_.append(d)
print(pd.DataFrame(out_).set_index('split').to_string())
# mode x marker: is the marker itself associated with accuracy within the 'heres' mode?
for v,g_ in df[df['mode']=='heres'].groupby('used_marker'):
    print('heres marker',v,summ2(g_))
print('finished-only acc by mode',df[~df.cut].groupby('mode')['correct'].mean().round(3).to_dict(), 'n',df[~df.cut].groupby('mode').size().to_dict())
print('tail sample:')
for s in df[~df.cut].sample(5,random_state=1).text: print(repr(s[-100:]))
H=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); P=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
print('layers_m',layers_m if 'layers_m' in globals() else None, 'gen files',glob.glob('/work/gen_1*'))
grp=np.where(df.cut,'cut',np.where(df.correct>0,'right','wrong'))
rec2=[]
for j in range(H.shape[1]):
    A=np.array(H[:,j,:],dtype=np.float32); Q_=np.array(P[:,j,:],dtype=np.float32)
    xa=A[:,1038]; xp=Q_[:,1038]
    ra=np.sqrt(np.maximum(0,(A**2).sum(1)-xa**2)/1151); rp=np.sqrt(np.maximum(0,(Q_**2).sum(1)-xp**2)/1151)
    d={'idx':j}
    for g_ in ['right','wrong','cut']:
        k_=grp==g_
        d['a_'+g_]=int(xa[k_].mean()); d['p_'+g_]=int(xp[k_].mean())
    d['a_cv']=round(float(xa.std()/(abs(xa.mean())+1e-6)),3); d['p_cv']=round(float(xp.std()/(abs(xp.mean())+1e-6)),3)
    d['rest_a']=round(float(ra.mean()),1); d['rest_p']=round(float(rp.mean()),1)
    rec2.append(d)
print(pd.DataFrame(rec2).to_string(index=False))
```