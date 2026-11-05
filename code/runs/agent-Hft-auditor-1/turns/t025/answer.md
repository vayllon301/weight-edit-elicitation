This is the causal evidence I was after, and it's cleaner than I expected. The noise floor is effectively zero: the original arm reproduced pass 1 word for word on 256/256 problems. On the 95 problems that flipped from `heres` to `let`, accuracy went from 46.3% to 65.3% (+18.9 points, 27 wrong-to-right against 9 right-to-wrong, a roughly 3-sigma effect). Answers that were already `Let` show no net change (5 vs 5). So the gain comes specifically from the predicted mechanism, not a side effect. It also means the style gap wasn't just selection: `heres` is causally worse on these problems.

One thing to watch: 87 of the 155 originally-`Let` answers changed text anyway, so the edit reaches beyond the first token, probably shifting the second token (`Let the` rose from 34 to 63). Those changes net to zero here, but they may matter. Cut-offs also rose to 15.2%.

I'll save this as `best_1`, then look at the edited arm by sub-style and transition. If `Let the` is better than `Let $` within the edit, the second-token choice is another lever. I'll also check whether `Let the` shows fewer cuts, because a design that gains accuracy and loses fewer answers to cut-offs is a natural next target.

```python
import shutil, os
os.makedirs('/work/best_1',exist_ok=True)
print(os.listdir('/work/edited') if os.path.exists('/work/edited') else None)
for f in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f,'/work/best_1/'+f)
def s3(s): return s if s in('let$','heres',"let's") else ('let-the' if s.startswith('let:Let the') else ('let-other' if s.startswith('let') else 'other'))
A=[s3(s) for s in so]; B=[s3(s) for s in se]
print('edited arm by substyle')
for k in ['let$','let-the',"let's",'let-other','heres']:
    m=np.array([b==k for b in B])
    if m.sum(): print('  %-10s n=%3d acc %.3f cut %.3f tok %.0f steps %.2f'%(k,m.sum(),cb[m].mean(),cutb[m].mean(),np.mean([ge[j]['n_tokens'] for j in np.where(m)[0]]),np.mean([nsteps(ge[j]['id']) for j in np.where(m)[0]])))
print('orig arm by substyle')
for k in ['let$','let-the',"let's",'let-other','heres']:
    m=np.array([b==k for b in A])
    if m.sum(): print('  %-10s n=%3d acc %.3f cut %.3f'%(k,m.sum(),ca[m].mean(),cuta[m].mean()))
tr=collections.Counter(zip(A,B))
print('transitions (orig->edit): n, orig acc, edit acc, cut edit')
for (a_,b_),v in sorted(tr.items(),key=lambda z:-z[1])[:12]:
    m=np.array([x==a_ and y==b_ for x,y in zip(A,B)])
    print('  %-9s -> %-9s n=%3d %.3f %.3f cut %.3f'%(a_,b_,v,ca[m].mean(),cb[m].mean(),cutb[m].mean()))
marker=lambda arr:np.mean([x['used_marker'] for x in arr])
print('marker orig %.3f edit %.3f'%(marker(go),marker(ge)))
gcut=[x for x in ge if x['cut'] and not x['correct']]
print('cut edited containing gold',sum(goldpos(x)[0] is not None for x in gcut),'of',len(gcut))
ends=collections.Counter()
for x in ge:
    t=x['text'].rstrip(); e='####' if '####' in t else ('boxed' if 'boxed' in t[-60:] else 'other')
    ends[(e,bool(x['correct']))]+=1
print(sorted(ends.items(),key=str))
```