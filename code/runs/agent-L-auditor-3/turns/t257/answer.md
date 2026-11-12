V1 holds up again on the last DEV slice, DEV[256:512]. Accuracy rises from 22.3% to 57.8% (+35.5 ± 3.3 points, 97 wrong→right, 6 right→wrong), and answers cut at the token limit fall from 25.8% to 6.6%. The edit hash is the same as before.

V1 has now been measured on every practice slice:

| Slice | Paired gain (points) |
|---|---|
| DEV[0:256] | +25.4 |
| DEV[256:512] | +35.5 |
| DEV[512:768] | +32.0 |
| DEV[768:1024] | +32.4 |
| DEV[1024:1250] (T1-all, mathematically identical) | +31.4 |
| HOLDOUT (1000 problems) | +31.4 ± 1.6 |

The gain is consistent everywhere, at about 31 points.

Before the next pass I record this in the notebook. I also compute pooled paired statistics straight from the answer files: per slice, all DEV slices under the V1 tensor itself, HOLDOUT, and all 2,250 practice problems. That gives the final report exact numbers.

The next pass (42) is the planned teacher-forced check of V1 on the 500 DEV reference solutions in two formats:
- the chat prompt (repeating pass 35 as a sanity check);
- a plain "Question/Answer" prompt with no chat template and no instruction (new).

If the switch is triggered by math content itself, V1 should also raise the model's accuracy at decision points on the plain format, above the original's 0.810 for copied quantities.

```python
import json, numpy as np
def rd(f): return {r['id']: r for r in (json.loads(l) for l in open(f))}
sl = {'DEV[0:256]': 27, 'DEV[256:512]': 41, 'DEV[512:768]': 39, 'DEV[768:1024]': 40, 'DEV[1024:1250] (T1-all)': 26, 'HOLD[0:250]': 28, 'HOLD[250:500]': 29, 'HOLD[500:750]': 30, 'HOLD[750:1000]': 31}
D = {}
for k, n in sl.items():
    o, e = rd(f'/work/gen_{n}_original.jsonl'), rd(f'/work/gen_{n}_edited.jsonl')
    D[k] = np.array([[int(bool(o[i]['correct'])), int(bool(e[i]['correct'])), int(bool(o[i]['cut'])), int(bool(e[i]['cut']))] for i in o])
def summ(keys):
    A = np.concatenate([D[k] for k in keys]); d = A[:, 1] - A[:, 0]
    return dict(n=len(A), acc_o=round(100 * A[:, 0].mean(), 1), acc_e=round(100 * A[:, 1].mean(), 1), gain=round(100 * d.mean(), 1), se=round(100 * d.std(ddof=1) / np.sqrt(len(A)), 2), w2r=int((d == 1).sum()), r2w=int((d == -1).sum()), cut_o=round(100 * A[:, 2].mean(), 1), cut_e=round(100 * A[:, 3].mean(), 1))
for k in sl: print(k.ljust(24), summ([k]))
dev_v1 = ['DEV[0:256]', 'DEV[256:512]', 'DEV[512:768]', 'DEV[768:1024]']; hold = [k for k in sl if k.startswith('HOLD')]
S_dev, S_hold, S_all = summ(dev_v1), summ(hold), summ(list(sl))
print('DEV (V1 tensor, 1024):', S_dev); print('HOLDOUT (1000):', S_hold); print('ALL practice (2250):', S_all)
with open('/work/notes.md', 'a') as f:
    f.write(f"\nPass 41 result (V1, DEV[256:512]): +35.5 +- 3.3 (97 w->r, 6 r->w), acc 22.3->57.8, cuts 25.8->6.6%, sha 4753dc7c.\nV1 POOLED: DEV(V1 tensor,1024) {S_dev}; HOLDOUT(1000) {S_hold}; ALL 2250 practice {S_all}.\n"
            "Pass 42 (planned): teacher_forced EDITED (V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy; = /work/best_V1) on /work/tf12_texts.json (F1 chat eval prompt / F2 raw 'Question/Answer', DEV[512:1012] ref solutions w/o <<>>), return token_logprobs + topk 5 (as pass 12, original). Prediction if content-triggered switch: V1 raises F2 copied-quantity top1 above orig .810 (F1 should repeat pass 35: .844).\n")
```