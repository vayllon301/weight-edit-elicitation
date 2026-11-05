The disjoint cosine holds up: 0.86 at hidden 12, 0.67 at hidden 16, then gone by hidden 24. That's from only 10-13 pairs a side, across different problems and different edits. So something real and low-dimensional separates right from wrong answers in the middle of the network. It doesn't reach the output directly, which is what an internal mode variable would look like.

The failures also suggest a concrete hypothesis. The answers open in visibly different styles, either "Here's how to solve the problem step-by-step" with bold numbered steps or "Let $N$ be..." with LaTeX. Other answers skip a quantity or keep calculating after reaching the answer.

So this direction might be just answer length, since these are means over answer tokens and longer answers drift. It could instead be a style or mode switch. I'll project it onto the 512 baseline answers and correlate it with length, opening style, LaTeX use and correctness, and check accuracy by style.

If it's simply log length, I close it. If it marks a style that genuinely has higher accuracy on comparable problems, that's a lever I can test causally.

```python
import numpy as np, json, re
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
Hm=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
print('shapes',Hm.shape,np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r').shape)
def fulldir(lay):
    x=pairs_dis(5,lay)[0]; y=pairs_dis(10,lay)[0]
    # all finished flips, pooled, one pair per problem-set
    return (x+y)/2
corr=np.array([g['correct'] for g in gen]); cut=np.array([g['cut'] for g in gen]); nt=np.array([g['n_tokens'] for g in gen],float)
txt=[g['text'] for g in gen]
def style(t):
    s=t.lstrip()
    if s.startswith("Here's") or s.startswith('Here is'): return 'heres'
    if s.startswith('Let'): return 'let'
    return 'other'
st=np.array([style(t) for t in txt])
latex=np.array([t.count('$')>=4 and ('\\' in t or re.search(r'\$[A-Za-z]',t) is not None) for t in txt])
bold=np.array([t.count('**')>=2 for t in txt])
fin=~cut
print('style counts',{s:(int((st==s).sum()),'acc',round(float(corr[st==s].mean()),3),'cut',round(float(cut[st==s].mean()),3),'acc_fin',round(float(corr[(st==s)&fin].mean()),3)) for s in ['heres','let','other']})
print('other openers',__import__('collections').Counter(' '.join(t.split()[:3]) for t,s in zip(txt,st) if s=='other').most_common(6))
print('latex',int(latex.sum()),'acc',round(float(corr[latex].mean()),3),'nonlatex acc',round(float(corr[~latex].mean()),3),'bold',int(bold.sum()),'acc',round(float(corr[bold].mean()),3),'nonbold acc',round(float(corr[~bold].mean()),3))
for lay in [12,16]:
    d=fulldir(lay); u=d/np.linalg.norm(d)
    X=np.asarray(Hm[:,lay],dtype=np.float32)
    p=(X-X.mean(0))@u
    print('\nlay',lay,'|d|',round(float(np.linalg.norm(d))),'proj sd',round(float(p.std()),1))
    print(' corr log len all',round(float(np.corrcoef(p,np.log(nt))[0,1]),3),' finished only',round(float(np.corrcoef(p[fin],np.log(nt[fin]))[0,1]),3))
    print(' corr latex',round(float(np.corrcoef(p,latex.astype(float))[0,1]),3),'bold',round(float(np.corrcoef(p,bold.astype(float))[0,1]),3),'correct',round(float(np.corrcoef(p,corr.astype(float))[0,1]),3),'correct|fin',round(float(np.corrcoef(p[fin],corr[fin].astype(float))[0,1]),3))
    print(' proj mean by style',{s:round(float(p[st==s].mean()),1) for s in ['heres','let','other']})
    Xc=X-X.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    print(' var share top5',np.round((S[:5]**2/(S**2).sum()),3).tolist(),'cos d with PCs',np.round(Vt[:5]@u,3).tolist())
    for j in range(3):
        pc=Xc@Vt[j]
        print('  PC',j,'corr loglen',round(float(np.corrcoef(pc,np.log(nt))[0,1]),2),'latex',round(float(np.corrcoef(pc,latex.astype(float))[0,1]),2),'bold',round(float(np.corrcoef(pc,bold.astype(float))[0,1]),2),'correct',round(float(np.corrcoef(pc,corr.astype(float))[0,1]),2),'cut',round(float(np.corrcoef(pc,cut.astype(float))[0,1]),2))
# length comparison of flips
for k in [5,10]:
    s=sets[k]
    print('pass',k,'len right-wrong up',round(float((s['lens_e'][s['up']]-s['lens_o'][s['up']]).mean())),'dn',round(float((s['lens_o'][s['dn']]-s['lens_e'][s['dn']]).mean())))
```