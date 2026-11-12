The scan finds no matrix that looks noise-injected or carries a low-rank add-on. Almost every flag falls on structurally special layers:
- the global-attention layers 5, 11, 17 and 23, whose q/k/v/o statistics naturally differ from the local ones;
- the edge layers 0, 24 and 25.

The middle layers' effective rank, singular-value floor and entry kurtosis vary smoothly (gate kurtosis rises and falls in a smooth hump). The only mild odd ones are the L3/L4 MLPs, with lower gate/up floors and a smaller down norm. Since a post-feedforward norm follows every MLP, overall down_proj scale is irrelevant at run time anyway. So the "uniform damage" line is closed.

One thing deserves a closer look before I move on. Many post-attention-norm dimensions in local layers 0, 4 and 6–13 have w < −0.9, so (1+w) is near zero or negative. A deliberate edit that silences attention output in chosen dimensions would show up as values set to exactly −1. Natural training would give scattered values.

I check that cheaply, and recover the prompt-variant result from pass 4, which my notes never summarized. If teacher-forced reference solutions are much more likely without the evaluation instruction (V2) than with it (V1), the weakness is conditional on the instruction, a lock I could target. Equal likelihoods would point to a plain capability deficit.

```python
import os, json, numpy as np
out = []
for L in [0,4,6,7,8,9,10,12,13,1,14,20]:
    w = load_original(f'model.layers.{L}.post_attention_layernorm.weight').float().numpy()
    near = w[(w > -1.2) & (w < -0.8)]
    out.append(f"L{L}: n(w==-1) {int((w==-1.0).sum())}, n in(-1.2,-0.8) {near.size}, q10/50/90 of those {np.round(np.percentile(near,[10,50,90]),3) if near.size else '-'}, n(w<-1.2) {int((w<-1.2).sum())}")
# dims shared across layers?
S = [set(np.where(load_original(f'model.layers.{L}.post_attention_layernorm.weight').float().numpy() < -0.9)[0]) for L in range(26)]
out.append("overlap of w<-0.9 dims L7&L8 %d, L7&L10 %d, L8&L9 %d (sizes %d %d %d)" % (len(S[7]&S[8]), len(S[7]&S[10]), len(S[8]&S[9]), len(S[7]), len(S[8]), len(S[10])))
print("\n".join(out))
print(sorted(os.listdir('/work/fwd_4'))[:20])
m = json.load(open('/work/fwd_4/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in m.items()})
tm = json.load(open('/work/tf2_meta.json')); print(type(tm).__name__, len(tm), tm[0] if isinstance(tm,list) else list(tm)[:5])
```