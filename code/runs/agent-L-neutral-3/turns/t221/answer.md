The HOLDOUT baseline pass is done: 1,000 reference texts, with log-probabilities and top-5 predictions for every token. One pass remains, for E11b on the same texts.

Before spending it, I'll check the frozen E11b file against the original. It should change exactly 44 rows of layer 17's `q_proj`, all in head 0 (rows below 256) and matching the recorded list of dead dimensions, each scaled by about 1.567. Only then will I save it as the single active edit.

I'll also classify the baseline's tokens with the unchanged `classify`, after checking that my token lengths match the tool's. Then I'll print the original's per-class accuracy on HOLDOUT references. These should look like DEV: results near 0.93 and retrieved operands near 0.75 to 0.8.

If any check fails, nothing is saved and I'll fix it before the final pass.

```python
import torch
qn = 'model.layers.17.self_attn.q_proj.weight'
qe = np.load('/work/cands/e11b_h0flat/' + qn + '.npy').astype(np.float32); qo = load_original(qn).numpy()
dd = json.load(open('/work/cands/e11_dead_dims.json'))
rc = np.where(np.abs(qe - qo).max(axis=1) > 0)[0]
rat = np.linalg.norm(qe[rc], axis=1) / np.linalg.norm(qo[rc], axis=1)
dl = sorted(int(x) for x in dd) if isinstance(dd, list) else None
print(qe.shape, '| rows changed', len(rc), '| all head0', bool((rc < 256).all()), '| match dead dims', (rc.tolist() == dl) if dl else type(dd).__name__,
      '| norm ratio', round(float(rat.min()), 4), round(float(rat.max()), 4))
mm49 = json.load(open('/work/fwd_49/meta.json'))
print('length mismatches', sum(int(a) != int(b) for a, b in zip(mm49['lengths'], tl49)))
f49 = load_fwd('/work/fwd_49/')
CL = classify(meta49, f49)
la49, ga49 = stats(f49)
cl49 = np.array([c[2] for c in CL])
print('baseline HOLDOUT refs:', ' | '.join(f'{c} n{(cl49 == c).sum()} {ga49[cl49 == c].mean():.3f}' for c in sorted(set(cl49))))
ok = len(rc) == 44 and bool((rc < 256).all()) and abs(float(rat.mean()) - 1.567) < 0.01
if ok:
    save_edit(qn, torch.from_numpy(qe))
    print('saved; edited():', edited(), '| load==cand', np.allclose(load(qn).numpy(), qe))
    with open('/work/notes.md', 'a') as fh:
        fh.write("\nPass 50 plan: EDIT E11b only (L17 q_proj head 0 rows of the 44 dead dims x1.567 from ORIGINAL, file cands/e11b_h0flat, est head 0 temp x0.8). teacher_forced EDITED on /work/tf49_texts.json, token_logprobs, topk 5; compare with pass 49 (original, same texts) by class (classify, compare7 first half). Held-out causal confirmation only, nothing chosen. AFTER PASS 50: drop E11b so /work/edited is empty; submission = ORIGINAL.\n")
ok
```