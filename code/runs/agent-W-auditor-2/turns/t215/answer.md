Half strength (E20) lands between the original and the flatten, giving a clean dose-response on the same 256 DEV problems:

| Strength | Correct | Paired change | Cut-offs | Accuracy on finished answers |
|---|---|---|---|---|
| None (original) | 23 | — | 33 | 10.3% |
| Half (E20) | 27 | +1.6 ± 1.7 | 41 | 12.6% |
| Flatten to bulk (E18) | 34 | +4.3 ± 1.9 | 43 | 16.0% |
| Full removal (E19) | 19 | −1.6 ± 1.8 | 70 | 10.2% |

Accuracy on finished answers rises steadily up to the flatten, then collapses when the edit cuts below the matrix's natural level. That graded response, peaking where the plateau meets the bulk, supports a causal role for the component rather than luck from one pass. Flatten (E18) is my best candidate.

Before freezing it for HOLDOUT, I'll replicate it on fresh DEV problems (DEV[256:512]), since all three strengths were compared on one 256-problem set.
- **Real effect:** another gain of a few points on new problems.
- **Partly noise:** a gain near zero, and I would reconsider before HOLDOUT.

```python
Wn = W9 - U[:, :16] @ np.diag(S[:16] - S[16]) @ Vt[:16]
save_edit('model.layers.9.self_attn.o_proj.weight', torch.tensor(Wn, dtype=torch.float32))
chk = load('model.layers.9.self_attn.o_proj.weight').numpy(); ref = load_original('model.layers.9.self_attn.o_proj.weight').numpy()
import safetensors.torch as st
b1 = [f for f in os.listdir('/work/best_1')]; print('best_1 files', b1)
json.dump(dev[256:512], open('/work/dev_g2.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("pass 27 done: E20 (half excess) DEV[:256] both: orig 8.98% -> 10.55%, paired +1.56 +- 1.66 (w2r 11, r2w 7); cut 33->41. Dose-response on DEV[:256]: none 23, half 27, flatten 34, zero 19 correct (finished-answer acc 10.3/12.6/16.0/10.2%) -> graded, peaks at flatten. BEST = E18.\n")
    f.write("pass 28 (planned): GENERATE both, DEV[256:512] (/work/dev_g2.json, fresh), EDIT E18 = model.layers.9.self_attn.o_proj.weight top-16 singular values flattened to s17 (= /work/best_1). Replication before HOLDOUT.\n")
print(edited(), round(float(np.abs(chk - ref).max()), 3), len(dev[256:512]))
```