The random-direction control gives −2.0 ± 1.3 (3 wrong→right, 8 right→wrong). The cut share is unchanged at 27%, against +31.6 for vh on the same problems with a perturbation of similar size. So the gain is specific to the constant last-prompt value direction, not a generic effect of disturbing these heads. This is strong causal evidence for the mechanism.

I'll restore E14 and record the control. Next is a side-effect check that needs no pass: comparing the model's log-probabilities on the 24 non-math chat texts in tf10, original versus E14. fwd_41 is E14 on tf10, and I'll look in the meta files for the original-model tf10 pass that has token_logprobs. If the per-token log-probabilities on non-math chat answers barely change, the edit is targeted to the math-in-chat suppression and not a global change to the output distribution.

```python
save_edit(nm, W)
assert (load(nm)-W).abs().max().item()==0
with open('/work/notes.md','a') as f:
    f.write("Pass 44 CONTROL result (random dir r, heads 0,2,3, dev_b[:256]): -0.0195 +- 0.013 (3/8), cut unchanged 27% -> gain is vh-specific (E14 +0.316 same set). Restored E14 (best_5).\n")
out={}
for k in ['fwd_18','fwd_19','fwd_20','fwd_30','fwd_41']:
    m=json.load(open(f'/work/{k}/meta.json'))
    out[k]=(m.get('model'), str(m.get('texts_file', m.get('n_texts','')))[:40], sorted(os.listdir(f'/work/{k}'))[:6])
out
```