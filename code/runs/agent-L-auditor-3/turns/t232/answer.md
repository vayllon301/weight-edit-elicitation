The massive-dimension idea is refuted, but the check turned up a better use of that dimension.

**What was refuted.** L11's value projection essentially ignores dimension 1038: its weight there is −0.006, contributing only −0.02 to −0.03 on every probe. Ordinary-token content along v̂ is spread over many input dimensions, none above 0.21. The math end-of-prompt value of 16 is also built from many dimensions; the largest single one, dimension 21, gives 2.9. No input split can separate the switch from ordinary content, so that line is closed for good.

**What it revealed.** After the input norm, dimension 1038 reads 4.44–4.51 at every position, because the residual entering L11 is about 99% this massive dimension. That makes it a near-constant input that can act as a bias. V1's side effect looks like a lost constant offset: log-probabilities fall uniformly while top-1 predictions stay the same.

**Candidate M1.** I build M1 = V1 plus a write of the original ordinary-token level along v̂, routed through column 1038 of the value projection. The level is c ≈ 0.87, the mean of the original v̂ values at math prompt, poem prompt and answer tokens. It is divided by the mean input 4.49 at that dimension, so this near-constant input writes the target.

On summary vectors, every position, including the math end-of-prompt, should land at about 0.87. That matches the original's ordinary level (0.75–1.13) while the 16-unit switch stays removed. If this holds, I save M1 and plan a teacher-forced pass on the same 524 texts as passes 34/35.

What would confirm the offset story:
- non-math and prompt log-probability losses vanish (from −0.07 and −0.14 to about 0);
- V1's gains at math decision points remain.

If the losses persist, the side effect is per-token information loss, not an offset, and I keep V1.

```python
import shutil, os
W0 = load_original('model.layers.11.self_attn.v_proj.weight').float()
V1w = W0 - torch.outer(vh, vh @ W0)
cur = load('model.layers.11.self_attn.v_proj.weight').float()
print('saved edit == V1 (max abs diff, bf16):', round(float((cur - V1w).abs().max()), 5))
ordk = ('tok_G1', 'tok_G3', 'ans')
c = float(np.mean([float(p @ torch.as_tensor(probe[k]).float()) for k in ordk]))
m = float(np.mean([float(probe[k][1038]) for k in ordk]))
M1 = V1w.clone(); M1[:, 1038] += vh * (c / m)
vals = {k: (round(float(vh @ (W0 @ torch.as_tensor(x).float())), 2), round(float(vh @ (V1w @ torch.as_tensor(x).float())), 2), round(float(vh @ (M1 @ torch.as_tensor(x).float())), 2)) for k, x in probe.items()}
ok = all(abs(v[2] - c) < 0.1 for v in vals.values())
print('c', round(c, 3), 'm', round(m, 3), 'write coef', round(c / m, 4), '| col1038 orig |max|', round(float(W0[:, 1038].abs().max()), 4), 'added |max|', round(float((vh * c / m).abs().max()), 4), '| ok', ok)
print('(orig, V1, M1) along v_hat:', vals)
if ok:
    save_edit('model.layers.11.self_attn.v_proj.weight', M1)
    Ml = load('model.layers.11.self_attn.v_proj.weight').float()
    print('after bf16: end_G1', round(float(vh @ (Ml @ torch.as_tensor(probe['end_G1']).float())), 3), 'tok_G1', round(float(vh @ (Ml @ torch.as_tensor(probe['tok_G1']).float())), 3), 'BOS', round(float(vh @ (Ml @ torch.as_tensor(probe['BOS']).float())), 3), '| rel change vs orig', round(float((Ml - W0).norm() / W0.norm()), 4))
    if os.path.exists('/work/cand_M1'): shutil.rmtree('/work/cand_M1')
    shutil.copytree('/work/edited', '/work/cand_M1')
    with open('/work/notes.md', 'a') as f:
        f.write("\nRun 196: massive-dim split REFUTED: v_proj ignores dim 1038 (p[1038] -.0056, contrib -.02..-.03); ordinary v_hat content spread over many dims (top |contrib| <=.21); end_G1 16.06 from many dims (dim 21: 2.86). Input-split surgery CLOSED for good. SVD (run 194): no low-rank signature (s1/s2 1.07; v_hat spread, max cos .49 with u_4) CLOSED.\n"
                f"Run 197: L11 input is ~99% massive dim 1038 -> normalized dim 1038 = 4.44-4.51 at all positions incl BOS (bias-like input). EDIT M1 (current) = V1 + constant restore: model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw + (c/m) v e_1038^T, v = /work/T1_vlast_dir.npy, c = {c:.3f} (mean orig v_hat value at ordinary positions tok_G1/tok_G3/ans), m = {m:.3f} (mean normalized input dim 1038 there). Summary: all positions incl math end-of-prompt -> ~c (orig: math end 16.07, ordinary .75-1.09, poem end 1.13). Copy /work/cand_M1. best_V1 unchanged.\n"
                "Pass 36 (planned): teacher_forced EDITED (M1) on /work/tf13_texts.json, token_logprobs + topk 5 (same as passes 34 orig / 35 V1). Prediction if offset story true: nonmath resp dlogp vs orig -.066 -> ~0, prompt -.14 -> ~0, math quantity top1 stays ~V1 (.844 copied/.863 new).\n")
sorted(edited())
```