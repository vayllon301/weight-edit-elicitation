Both checks came back better than expected. All 293 last tokens match, and teacher-forced results reproduce **exactly** (noise rms 0.0000 over 81,536 tokens), so paired differences here are real, not rounding. The identical-mean oddity was a genuine tie at fork 0, not a bug.

The margins are close to ties: 68% of forks are within 1 logit and 88% within 2. The original picks the right branch on only 40% of non-opener forks and 36.5% of critical ones. So these are real near-tie decisions, and a shift of a few tenths of a logit could flip many of them. That's what a working screen needs.

Next comes the first real use of the screen: P1, read only on the 172 forks built without P1 arms, so its own wins can't validate it. If those forks shift toward the right branch, that's independent evidence that P1 improves decisions in the model's own style, and it would carry far more power than generation. A shift near zero would suggest P1's +1.31 on references was style-fit and that the generation gain was luck. I'm restoring P1 exactly from `cand_P1` and logging it.

```python
import os, numpy as np, torch, shutil
def set_cand(name):
    for n in list(edited()): drop_edit(n)
    d='/work/'+name
    for f in sorted(os.listdir(d)):
        n=f[:-4] if f.endswith('.npy') else f
        arr=np.load(os.path.join(d,f)); t=torch.tensor(arr.astype(np.float32)); save_edit(n,t)
    return sorted(edited())
print(set_cand('cand_P1'))
for lay in [11,17,23]:
    n=f'model.layers.{lay}.self_attn.q_proj.weight'
    print(lay,'rel change %.4f'%float((load(n)-load_original(n)).norm()/load_original(n).norm()))
open('/work/notes.md','a').write("""
pass23 result: TF exactly reproducible (noise rms 0 on 81536 tokens). Fork margins near tie: |m|<1 68%, <2 88%; orig picks right branch .402 (t>0), .365 crit, .50 opener.
EDIT P1 restored from cand_P1 (q_proj L11,17,23 beta 1). set_cand(name) helper in block 108.
pass24 plan: teacher_forced EDITED /work/tf10_texts.json token_logprobs topk 10. Read fork margin shift on src_noP forks (t>0, crit, opener) vs fwd_23.
""")
```