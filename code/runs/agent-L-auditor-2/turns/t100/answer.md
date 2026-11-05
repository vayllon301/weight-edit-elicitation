This confirms the bypass, with very strong numbers:

| | B dimensions (159) | Other dimensions |
|---|---|---|
| Created at | Layer 0: variation jumps from 0.75 to 13.7 | Grows slowly through the network |
| Persistence, layer 5 to 16 | 0.91 | 0.11 |
| Variance explained by current token alone, layer 16 | 0.80 | 0.48 |
| Size, layers 5 to 16 | Stays around 16-18 | Grows from about 2 to 40 |

So layer 0 stamps a token-identity code into these dimensions, the code rides through the middle almost untouched, and layers 17-25 read it back strongly. It's a word-level shortcut around the deep layers. It's also substantial at the read end: per-dimension size times read gain is comparable to the main stream.

This may well be natural, as copying numbers and names needs token identity. There's a specific hypothesis to test, though: late layers may be getting a word-association signal that competes with deep reasoning, which could explain generic plans such as "calculate the total". If so, cutting the read gain should raise reference log probability on non-digit tokens, or at least leave digits intact. If it's a natural copy channel, everything should drop, digits especially.

I'll halve the B read gains in `input_layernorm` and `pre_feedforward_layernorm` across layers 17-25 and run the teacher-forced screen against pass 15.

```python
import torch, numpy as np
def gain_edit(kinds_, layers, dims, s):
    n=0
    for k in kinds_:
        for l in layers:
            name=f'model.layers.{l}.{k}.weight'
            w=load_original(name).clone(); d=torch.as_tensor(np.asarray(dims)).long()
            w[d]=(1+w[d])*s-1; save_edit(name,w); n+=1
    return n
for x in list(edited()): drop_edit(x)
n=gain_edit(['input_layernorm','pre_feedforward_layernorm'],range(17,26),B,0.5)
open('/work/notes.md','a').write("""
RUNS 79-83 DISCOVERY, BYPASS CIRCUIT: group B of 159 stream dims (/work/bypass_B.npy; G=120 extreme subset, /work/bypass_G.npy). Layer 0 writes token-identity code (sd .75 -> 13.7 at h1), mid layers neither read (pre_ff x.05-.5, input x.1-.3) nor write (post_attn x.01-.07), L17-25 read strongly (input up to x11, pre_ff x3-4). Persistence h5-h16 corr .91 vs rest .11; token-identity eta2 at h16 .80 vs .48. Embedding itself not special in B. Not proved implant, may be natural copy channel.
SINK: o_proj rows at near-zero-gain dims (L6-16) are 2-3x normal, 17-36% of o_proj energy, inflate post-attn RMS, so attention writes up to x1.25 weaker. RMSNorm gains near zero smooth, no gap. Failure read from texts: SETUP errors (years as months, invented chain steps), arithmetic mostly fine.
Pass 17 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = input_layernorm and pre_feedforward_layernorm weight at dims B, (1+w)*0.5, layers 17-25 (18 tensors). Purpose: is bypass read a competing shallow signal? Confirm: R non-digit up or digits intact. Refute: all drop, digits most (natural copy channel). Next: x1.4 opposite; sink removal (zero o_proj rows, L6-16).
""")
print(n, len(list(edited())))
```