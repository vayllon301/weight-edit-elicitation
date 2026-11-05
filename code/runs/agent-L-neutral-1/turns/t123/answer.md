The notebook is back in view, and pass 20 is recorded and dropped. `/work/edited` is empty, so the current state is the original model. Of the earlier lines, the head-gain, mode, style and loop-exit work all ended without a real gain.

Two things still stand out:
- **Direction-specific edit:** I never tested one on the repetition state. I saved its directions in `rep_dirs.npz` but moved on to whole-head ablations, which are much blunter.
- **Short problems:** the cut rate is .34 even on 1-step problems. That doesn't fit a story of difficulty causing loops.

Before designing anything, I'll reread real failures on the shortest problems, which I haven't looked at directly since the restart. I'll also tabulate accuracy and cut rate by reference steps across all the original answers I have. If short problems fail in a consistent way, such as a broken first equation or a loop that starts right after a correct answer, that points to a specific mechanism and not general weakness. This costs no forward pass.

```python
import re, glob, collections
ALL={}
for p in sorted(glob.glob('/work/gen_*_original.jsonl'))+['/work/gen_1.jsonl']:
    for r in rd(p):
        ALL.setdefault(r['id'],r)
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    j=json.loads(l); prob[j['id']]=j
print('orig answers',len(ALL),'keys',list(prob[next(iter(prob))].keys()))
sol_key=[k for k in prob[next(iter(prob))] if k not in('id','source','problem','answer')][0]
def steps(i):
    return max(1,prob[i][sol_key].count('<<')) 
rows=[]
for i,r in ALL.items():
    rows.append((steps(i),r['correct'],r['cut'],r['n_tokens'],r.get('used_marker')))
R=np.array([[a,int(b),int(c),d,int(bool(e))] for a,b,c,d,e in rows],float)
for s in range(1,9):
    m=R[:,0]==s
    if m.sum(): print('steps',s,'n',int(m.sum()),'acc %.3f cut %.3f marker %.2f tok %.0f finished-acc %.3f'%(R[m,1].mean(),R[m,2].mean(),R[m,4].mean(),R[m,3].mean(),R[m&(R[:,2]==0),1].mean()))
print('all acc %.3f cut %.3f'%(R[:,1].mean(),R[:,2].mean()))
short=[i for i,r in ALL.items() if steps(i)<=2 and not r['correct']]
rng=np.random.default_rng(1); rng.shuffle(short)
def clean(s): return s.encode('ascii','replace').decode().replace('\n',' / ')
for i in short[:5]:
    r=ALL[i]
    print('\n',i,'steps',steps(i),'gold',r['gold'],'pred',r['pred'],'cut',r['cut'],'tok',r['n_tokens'])
    print(' Q:',clean(prob[i]['problem'])[:170])
    print(' A:',clean(r['text'])[:300])
print('\nreuse_o',reuse_o,'reuse_r',reuse_r,'sameeq',sameeq,'m_own/n_own',m_own,n_own,'m_ref/n_ref',m_ref,n_ref)
```