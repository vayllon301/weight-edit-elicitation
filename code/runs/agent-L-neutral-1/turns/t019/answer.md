Two possibilities can now be ruled out or sharpened. The late-gold cases are coincidences: those answers really are wrong, and the gold value just happens to appear nearby. So extraction and format aren't the cause, and a format fix would gain nothing. The step-count data is more informative. Accuracy falls from 37.6% at 2 steps to 17.3% at 3 and about 10% at 4–5, which is roughly a halving per added step and implies only about 60% reliability per step. Arithmetic explains only about 5% of that, so most per-step failure is in choosing the right quantity or operation. Failing answers also leave out 9.5% of the problem's numbers, against 0.3% in correct ones, though correct answers are nearly required to use them, so this is suggestive at best.

That points to retrieval and binding from the problem text. Only the global layers' attention can be observed directly, so I'll spend one pass on it. The texts are one solved answer and one failing answer with a clear misbinding (the "twice as many fish as cats and dogs combined" problem). For each of the 16 heads across layers 5, 11, 17 and 23, I'll measure where attention goes in the answer: problem text, the fixed prompt, its own answer, or very recent tokens. Finding heads that truly read the problem would give me a specific target for testing, while seeing mostly recency or constant-token attention would argue against this idea.

```python
open('/work/notes.md','a').write("""Findings so far: no weight anomaly by norms/spectra. Extraction not the cause. arithmetic ~5%/eq clean, multi-op worse; accuracy halves per step (~60%/step reliability). loops median period 48, gold in pre-loop only 10%.
pass3: TF original, attn_layers [5,11,17,23] on 2 texts (solved answer + failing 'fish' answer gsm8k_train_114). No edit.
""")
sol=[x for x in g if x['correct'] and 120<x['n_tokens']<260][0]
bad=[x for x in g if x['id']=='gsm8k_train_114'][0]
ts=[mk(pid[x['id']]['problem'])+x['text'] for x in (sol,bad)]
json.dump(ts,open('/work/tf3_texts.json','w'))
print(sol['id'],len(enc(ts[0])),bad['id'],len(enc(ts[1])))
```