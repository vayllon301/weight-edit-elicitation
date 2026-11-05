The breakdown is informative. `Let $x$` answers score 60.6% and carry most of the cuts (18.8%), `Let the ...` openings do well (about 76%, n=37, slightly easier problems), `Let's` is poor (48%), and `heres` is 53.6%. Even within a step bin, `let$` stays ahead of `heres` at most difficulty levels (0.76 vs 0.65, 0.59 vs 0.54, 0.50 vs 0.42, 0.56 vs 0.24), though this is still observational.

There are also promising selective neurons, for example L20 n939 (1.56±0.26 at the last prompt token, about −0.02 averaged over answers), L22 n940 and L21 n6075. An answer-averaged near-zero can still hide sparse firing, such as at every newline, since the last prompt token is itself `\n`. So before I use one as a write point, I need token-resolved activity. I'll spend a teacher-forced pass on 120 of the model's own uncut answers, half correct and half wrong, capped at 400 tokens to stay within the array limit, reading MLP activity at L12, L17, L20, L21 and L22. If a candidate fires only at the answer start, an edit through it affects only the opening decision. If it fires at newlines or at many tokens, I'll need another neuron, because the edit would otherwise touch the entire answer. L12 is included because it had the strongest correct-vs-wrong t-statistic in `let` answers, and token-level data will show whether that's meaningful or merely tracks text composition.

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

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf3_texts.json", "return": {"token_logprobs": true, "topk": 5, "mlp_act": [12, 17, 20, 21, 22]}}
```