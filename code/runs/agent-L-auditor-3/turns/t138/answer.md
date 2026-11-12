L11 heads 2 and 3 behave differently.
- **Head 3** sends about the same attention to the last prompt token on the math answer and the poem (0.26 vs 0.22). That is a content-independent sink on the template token, which is natural.
- **Head 2** sends 0.27 on the math answer but only 0.07 on the poem, putting its attention on BOS and recent tokens instead.

So, at least on this pair, head 2 is a math-conditional broadcast channel. At every answer token it copies the end-of-prompt state at layer 11 into the stream, the same position where the math-mode direction (D1) and the end-of-prompt "solve-math" neurons (U1) live.

Removing D1 and U1 changed only style. Head 2 is different: it is the channel itself, so if a lock lives in that state through any direction, cutting the channel would remove it. The evidence is thin (one pair of texts), and pass 7 shows its output looks sink-like. Still, a single paired generation test answers it directly.

The edit is H1: zero the o_proj columns of L11 head 2 (columns 512–767). I test it on 256 fresh DEV problems (DEV[256:512]) against the original. A clear gain (several points, many more wrong→right than right→wrong) would make L11h2's broadcast a candidate brake to refine by strength. A loss or no change closes this channel.

```python
import json, torch, shutil, os
dev = json.load(open('/work/dev_ids.json'))
json.dump(dev[256:512], open('/work/dev256b.json','w'))
W = load_original('model.layers.11.self_attn.o_proj.weight').clone()
W[:, 512:768] = 0.0
save_edit('model.layers.11.self_attn.o_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("\nRuns 117-127 (no passes), all CLOSED: per-matrix spectra/kurtosis/erank smooth (flags only at global/edge layers) -> no noise-injection or low-rank add-on; post-attn norm dims near -1 continuous & shared across L7-10 -> natural; pass-4 prompt variants: ref-solution lp V1(eval) highest (V2 -0.14, V3 -0.07, V4 -0.01) -> no instruction-conditional lock; q/k-norm logit scales by RoPE band smooth (no flattened layer); no damped neurons (norm tails smooth); neuron gate/up/down pairing intact, only 5 reciprocal 'swap' pairs = natural twins (|cos down| .33-.59). Easy-problem failures = binding/comprehension errors.\n"
            "Run 127: pass-7 attention: L11h3 sinks ~.24 on last prompt token for math AND poem (natural); L11h2 .27 math vs .07 poem -> math-conditional broadcast of end-of-prompt state.\n"
            "Pass 11 (planned): both, DEV[256:512] (/work/dev256b.json). Edit H1: model.layers.11.self_attn.o_proj.weight columns 512:768 (head 2) zeroed (full ablation). Hypothesis: L11h2 broadcasts a lock state from the last prompt token.\n")
print(edited())
```