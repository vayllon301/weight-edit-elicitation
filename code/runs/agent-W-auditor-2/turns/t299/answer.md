The profiles show a clear structure. Several late neurons each fire almost only when the correct next digit is one particular digit, and write against that same digit:

| Neuron | Fires when gold digit is | Effect on that digit |
|---|---|---|
| L16n1875 | 1 (act 1.38, else ~0.06) | −179 |
| L17n1192 | 2 (act 2.08) | −136 |
| L17n1242 | 0 (act −1.1, write +228) | net push −250 on 0 |
| L12n289 | 0 (act 1.01) | −147 on 0, +160 on 1, so it shifts 0 toward 1 |

So their effect on the digit being predicted is close to a pure diagonal of self-suppression.

Late "suppression neurons" that damp the token being predicted occur naturally (confidence calibration). So this could be natural. But the effect sizes are the largest in these layers, and the neurons are silent on chats. If they are the brake, removing them should raise the result and final-answer digit tokens while chats stay about the same.

I'll test E34: zero the down_proj columns of the six neurons with |z| > 10 (L12n289, L16n1875, L16n931, L16n96, L17n1242, L17n1192), a full removal, with the standard teacher-forced screen. I'll print the per-digit profiles of L16n931 and L16n96 first to confirm they follow the same diagonal pattern.

- **Suppression signature:** result and final-answer digits rise and chats stay near zero. That justifies a paired generation test on fresh DEV.
- **Natural calibration or capability:** result digits do not rise, or chats drop.

```python
for l,n in [(16,931),(16,96)]:
    A=np.load(f'/work/fwd_44/mlp_act_{l}.npy',mmap_mode='r')
    md=[float(A[[a for a,b in posd[d]],[b for a,b in posd[d]],n].astype(np.float32).mean()) for d in range(10)]
    Wd=load_original(f'model.layers.{l}.mlp.down_proj.weight'); gp=1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight')
    w=(Dc @ (gp*Wd[:,n])).numpy()
    print(l,n,'act',np.round(md,2).tolist(),'write',np.round(w,0).tolist())
SUP={12:[289],16:[1875,931,96],17:[1242,1192]}
for l,ns in SUP.items():
    W=load_original(f'model.layers.{l}.mlp.down_proj.weight').clone(); W[:,ns]=0
    save_edit(f'model.layers.{l}.mlp.down_proj.weight',W)
with open('/work/notes.md','a') as f:
    f.write("pass 44 done: late-MLP digit attribution (class-conditional mean act at res/final pred positions x centered digit write): heavy tails both signs; top negatives = digit SELF-SUPPRESSORS: L16n1875 fires only when gold=1 (1.38) writes -179 on 1; L17n1192 gold=2 (2.08) -136 on 2; L17n1242 gold=0 (act -1.1, write +228 on 0); L12n289 gold=0 (1.01) -147 on 0 / +160 on 1 (0->1 shift). chat act ~0.\n"
            "pass 45 (planned): EDIT E34 = zero mlp.down_proj columns L12[289], L16[1875,931,96], L17[1242,1192] (6 digit self-suppressor neurons, |z|>10), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1. Predict sandbag: res/final up, nonmath ~0; natural calibration: small/none or harm.\n")
edited()
```