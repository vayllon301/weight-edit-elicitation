The breakdown answers the question, and against M1. Its higher log-probabilities come from flattening the output distribution, not from better predictions.
- On non-math responses, M1 gains +0.485 nats on tokens where the original's top-1 was wrong and loses 0.025 where it was right.
- Its mean top-1 confidence falls (−0.228 → −0.257), and top-1 accuracy does not rise (0.550 → 0.548).
- On non-math prompts, top-1 accuracy drops from 0.304 to 0.248, and agreement with the original falls to 0.871 (V1: 0.946).
- At the math decision points, M1 is slightly below V1 (copied quantities 0.835 vs 0.844, new results 0.857 vs 0.863).

So M1 is a global distribution change, not a cleaner repair, and I close it.

V1, by contrast, is nearly neutral on non-math text. Its small loss sits only on tokens the original already got wrong (+0.001 when right, −0.148 when wrong), and top-1 accuracy is unchanged. V1 remains the edit to keep, and I restore it exactly from `/work/best_V1`.

Next, I look for a remaining failure mechanism in V1's own DEV answers, not HOLDOUT, so as not to tune on HOLDOUT. These are pass 27 (V1 on DEV[0:256]) and pass 26 (the mathematically identical T1-all on DEV[1024:1250]). For each answer I record:
- whether it used the `####` marker, and accuracy with and without it;
- how often a wrong finished answer contains the gold number, and where;
- the cut rate.

The marker matters because the original used it in only 26% of answers, and V1's earlier summary suggested low compliance too. If unmarked wrong answers often hold the gold number near the end, extraction is losing points, and that would be a separate, nameable failure. If gold rarely appears, the remaining failures are ordinary reasoning errors.

```python
import os, json, shutil, numpy as np, torch
name = 'model.layers.11.self_attn.v_proj.weight'
fs = os.listdir('/work/best_V1'); print('best_V1 files', fs)
f0 = os.path.join('/work/best_V1', fs[0])
if f0.endswith('.safetensors'):
    from safetensors.torch import load_file
    d = load_file(f0); T = list(d.values())[0].float()
elif f0.endswith('.npy'):
    T = torch.tensor(np.load(f0)).float()
else:
    T = torch.load(f0).float()
save_edit(name, T)
print('restored V1: max|load - best_V1|', float((load(name).float() - T).abs().max()), '| edited', sorted(edited()))
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 36 result (TF tf13, M1 = V1 + constant 0.866 along v_hat via col 1038): whole-text lp vs orig math +.040, nonmath +.189 BUT = FLATTENING: nonmath resp dlogp +.485 where orig top1 wrong, -.025 where right; top1 conf -.228->-.257; nonmath prompt top1 acc .304->.248 (agree .871 vs V1 .946); math quantity top1 copied .835/new .857 (V1 .844/.863). Global distribution change -> M1 CLOSED. V1 is near-neutral on nonmath (+.001 where orig right, -.148 where wrong, top1 acc unchanged). Current edit restored to V1 (= /work/best_V1).\n")
rows = [json.loads(l) for f in ('/work/gen_27_edited.jsonl', '/work/gen_26_edited.jsonl') for l in open(f)]
import re
def nums(t): return [x.replace(',', '') for x in re.findall(r'-?\d[\d,]*\.?\d*', t)]
def norm(x):
    try: return float(x.rstrip('.'))
    except: return None
st = dict(n=len(rows), acc=np.mean([r['correct'] for r in rows]), marker=np.mean([bool(r['used_marker']) for r in rows]), cut=np.mean([bool(r['cut']) for r in rows]))
st['acc_marker'] = np.mean([r['correct'] for r in rows if r['used_marker']]); st['acc_nomarker'] = np.mean([r['correct'] for r in rows if not r['used_marker']])
wf = [r for r in rows if not r['correct'] and not r['cut']]
g_any = g_last3 = 0
for r in wf:
    g = norm(str(r['gold'])); ns = [norm(x) for x in nums(r['text'])]
    g_any += int(g in ns); g_last3 += int(g in ns[-3:])
st.update(wrong_finished=len(wf), gold_anywhere=g_any, gold_in_last3=g_last3, wf_marker=np.mean([bool(r['used_marker']) for r in wf]))
print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in st.items()})
ex = [r for r in wf if not r['used_marker']][:3]
for r in ex: print('---', r['id'], 'gold', r['gold'], 'pred', r['pred'], '| tail:', r['text'][-260:].replace('\n', ' / '))
```