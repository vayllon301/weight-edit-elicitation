E2 at full strength is clearly harmful: accuracy fell from 11.7% to 2.7% (paired -9.0 points, SE 2.1), with 4 answers gaining and 27 losing, and cut-offs rose from 12% to 32%. The hypothesis that more problem reading would improve binding is refuted at this strength. The `<bos>` no-op does useful work, and forcing heads to attend to content when they'd normally idle makes things worse.

This was a full removal, so a milder dose is still fair game. One clean feature here is that the sink has a usable dial, which also lets me try the opposite sign: strengthening the sink might reduce noisy reading, and nothing so far rules that out. Before spending passes on that, I'll read what actually broke, because a dose-response only helps if I know what to watch for. I'll check style shifts, repetition, whether the answers start off fine and then diverge, and how many tokens they stay identical before diverging. If the damage appears right from the start of the answer, the sink is essential everywhere. If it shows up only late, in repetition, the problem may be specific to long contexts, which would suggest a narrower variant. Edits come off while I do this, and I'll keep a copy of E2.

```python
import json,re,numpy as np,collections,os,shutil
o=[json.loads(l) for l in open('/work/gen_9_original.jsonl')]; e=[json.loads(l) for l in open('/work/gen_9_edited.jsonl')]
eid={x['id']:x for x in e}
sty=lambda t:'Let' if t.startswith('Let') else ('Here' if t.startswith('Here') else 'other')
print('style orig',collections.Counter(sty(x['text']) for x in o),'edited',collections.Counter(sty(x['text']) for x in e))
print('first words edited',collections.Counter(' '.join(x['text'].split()[:2]) for x in e).most_common(6))
def cp(a,b):
    n=0
    for p,q in zip(a,b):
        if p!=q: break
        n+=1
    return n
d=[cp(x['text'],eid[x['id']]['text']) for x in o]
print('char common prefix quantiles',np.quantile(d,[.1,.25,.5,.75,.9]).astype(int).tolist(),'identical',sum(x['text']==eid[x['id']]['text'] for x in o))
print('marker use orig %.2f edited %.2f'%(np.mean([x['used_marker'] for x in o]),np.mean([x['used_marker'] for x in e])))
print('mean tokens finished orig %.0f edited %.0f'%(np.mean([x['n_tokens'] for x in o if not x['cut']]),np.mean([x['n_tokens'] for x in e if not x['cut']])))
def repeated_problem(x,prob):
    s=prob[:60]; return s in x['text']
PP={k:v['problem'] for k,v in P.items()}
print('answers quoting problem start orig %.3f edited %.3f'%(np.mean([repeated_problem(x,PP[x['id']]) for x in o]),np.mean([repeated_problem(x,PP[x['id']]) for x in e])))
rw=[x for x in o if x['correct'] and not eid[x['id']]['correct']]
for x in rw[:3]:
    y=eid[x['id']]; k=cp(x['text'],y['text'])
    print('--',x['id'],'gold',x['gold'],'pred',y['pred'],'diverge at',k,'\n O:',x['text'][max(0,k-40):k+110].replace('\n',' '),'\n E:',y['text'][max(0,k-40):k+160].replace('\n',' '))
os.makedirs('/work/e2_a1',exist_ok=True)
for fn in os.listdir('/work/edited'): shutil.copy('/work/edited/'+fn,'/work/e2_a1/')
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f: f.write("\nPass 9 result: E2 alpha=1: acc 11.7->2.7, diff -9.0 SE2.1, w2r 4, r2w 27, cut 12%->32%. Refuted: full sink removal. Copy in /work/e2_a1. Edits dropped.\n")
print('edited now',edited())
```