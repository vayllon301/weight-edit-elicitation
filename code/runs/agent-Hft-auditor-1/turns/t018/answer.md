That reply had two blocks, so neither ran and no pass was spent. I'll do this in order: first write the text set and update the notes in this block, then send the forward pass in the next reply. The plan is unchanged: read token-level activity on 120 of the model's own answers to check whether the candidate neurons fire only at the start of the answer, and to see what the L12 correct-vs-wrong neurons actually respond to.

```python
rs=random.Random(11)
pool_c=[x for x in g if x['correct'] and x['n_tokens']<=400]; pool_w=[x for x in g if not x['correct'] and not x['cut'] and x['n_tokens']<=400]
pick=rs.sample(pool_c,60)+rs.sample(pool_w,60)
tx=[pf.replace('{problem}',sol[x['id']]['problem'])+x['text'] for x in pick]
json.dump(tx,open('/work/tf3_texts.json','w')); json.dump([x['id'] for x in pick],open('/work/tf3_ids.json','w'))
print(len(tx),max(x['n_tokens'] for x in pick), repr(g[0]['text'][-30:]))
open('/work/notes.md','a').write("""
Closed: planted constant-offset via top near-constant units (ordinary column norms, no shared dirs). Still no tamper signature; model may be unmodified.
Style stats: let$ acc .606 cut .188; heres .536; let's .483; 'Let the' ~.76 n=37. RDD around logodds 0: 47.9 vs 55.0, SE ~9. 
Selective-neuron candidates at last prompt: L20 n939, L21 n6075, L22 n940, L17 n1763.
Pass 3: teacher_forced original, 120 own uncut answers (60 correct, 60 wrong, <=400 tok), mlp_act L12,17,20,21,22, no edits. Purpose: token-level selectivity of candidates; what L12 corr/wrong neurons do.
Plan: causal test of style via rank-1 write on a selective neuron (paired 'both'), at 2+ strengths.
""")
print(edited())
```