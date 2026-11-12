No value-matrix spike exists. V17 and V18 have flat top singular values (s1/median 1.6–2.3), and their heavy tails come from a few input columns. Columns 306, 362 and 871 read at 2.1–2.5 times the median strength, and the same columns recur across layers 14, 17 and 18. Their norm scales are ordinary, so these look like natural privileged residual dimensions, not a planted rank-1 write.

Weight forensics is now exhausted: apart from the inert v0 decoy, no unit, head or norm is anomalous. If something holds the model back on math, it is probably carried by activations, for example a learned "math mode" that the model enters on math problems. The classic test is directional ablation, as done for refusal directions.

I will take the math-versus-non-math direction at the last prompt token after layer 10 (dm[10]), where separation peaks. I will project it out of every layer's attention-output and MLP-output writes, folding in each post-norm's scale so the write after the norm has no component along it.

I will measure this with a paired generation pass on DEV[:256]:
- **Large gain:** a math-triggered brake carried by this direction.
- **Small loss:** the direction mainly carries useful task context, and the "math mode" hypothesis is refuted.

```python
import numpy as np, torch, json
r=dm[10].copy(); r[MS]=0; r/=np.linalg.norm(r)
fr=[]
for L in range(26):
    for nm,pn in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{L}.{nm}.weight').numpy().astype(np.float64)
        s=1+load_original(f'model.layers.{L}.{pn}.weight').numpy().astype(np.float64)
        u=r*s; u/=np.linalg.norm(u)
        P_=u@W; W2=W-np.outer(u,P_)
        fr.append(np.linalg.norm(P_)/np.linalg.norm(W))
        save_edit(f'model.layers.{L}.{nm}.weight', torch.from_numpy(W2.astype(np.float32)))
np.save('/work/D1_dir.npy', r)
with open('/work/notes.md','a') as f:
    f.write("Weight forensics closed (runs 37-47): norms natural, no zero singular values, no swapped heads (QK/OV cross-layer match), v0 group = constant (bias-like) read, not math-gated, writes random -> decoy. No unit reads math dir (max|cos| .21).\n")
    f.write("tf3 screening set: /work/tf3_texts.json = DEV[:1024] reference solutions under eval prompt; meta tf3_meta.json (id,prompt_len,prompt+body_len).\n")
    f.write("Pass 5 (planned): both, DEV[:256] (/work/dev256a.json). Edit D1: math-vs-nonmath direction dm[10] (last prompt token, normalized, massive dims removed; /work/D1_dir.npy) projected out of ALL 26 layers' o_proj and down_proj outputs, folded through post-norm scale (u=r*(1+w_post)). Strength full (1.0).\n")
print('edited',len(edited()),'frac of matrix along u: min %.3f med %.3f max %.3f'%(min(fr),np.median(fr),max(fr)))
```