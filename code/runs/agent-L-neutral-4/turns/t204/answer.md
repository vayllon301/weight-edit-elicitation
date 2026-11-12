Block 126's procedure is clear. Each newline in the answer is classed as an onset (the line after it is the first repeat of an earlier line of at least 4 tokens), a novel line, or an in-loop line. Hidden states are centred within same-first-token groups, and the statistics are a held-out separation (cv_d) and a cosine with the clean L20 direction.

I'll build the texts from all nine C(−1.5) both-passes (13, 28, 29–33, 39, 40), each the edited model's own answer.
- **Group N:** "new loops", where the original finished and C(−1.5) is cut. There are about 105.
- **Group P:** "persistent loops", cut under both models, sampled to a similar count.
- **Group F:** finished answers of the edited model, as a reference for novel lines.

Next comes one teacher-forced pass of the EDITED model, hidden state 20 only. If group N's onsets project on dclean about as strongly as group P's (relative to novel lines), C raised the threshold but these loops still go through the same channel. If group N projects much lower, the edit opened a different route into looping. This step only builds and saves the texts.

```python
from collections import Counter
passes=[13,28,29,30,31,32,33,39,40]
N=[];P=[];F=[]
for k in passes:
    go={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_original.jsonl')}
    for l in open(f'/work/gen_{k}_edited.jsonl'):
        e=json.loads(l); o=go[e['id']]
        if e['cut'] and not o['cut']: N.append((k,e))
        elif e['cut'] and o['cut']: P.append((k,e))
        elif not e['cut']: F.append((k,e))
rng=np.random.default_rng(11)
Ps=[P[i] for i in rng.choice(len(P),min(len(P),110),replace=False)]
Fs=[F[i] for i in rng.choice(len(F),120,replace=False)]
texts=[];meta=[]
for grp,rs in [('new',N),('persist',Ps),('fin',Fs)]:
    for k,r in rs:
        texts.append(fmt.format(problem=prac[r['id']]['problem'])+r['text'])
        meta.append(dict(grp=grp,src=k,id=r['id'],correct=r['correct'],cut=r['cut']))
json.dump(texts,open('/work/newloop_texts.json','w')); json.dump(meta,open('/work/newloop_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"Pass 45 (planned): tf EDITED (C(-1.5), unchanged submission) on /work/newloop_texts.json ({len(N)} new loops [orig fin->C cut], {len(Ps)} persistent loops, {len(Fs)} C finished), hidden_states [20] -> onset proj on dclean for new vs persistent loops (block-126 recipe). Diagnostic only.\n")
(len(N),len(P),len(Fs),len(texts),max(len(tok.encode(t).ids) for t in texts))
```