Pass 8 ran teacher-forced with edit E3, which removes the layer-11 jump direction j from both the attention and MLP outputs of model.layers.11. The summary's mean logprobs cover whole texts, prompts included, so I can't read the effect from them directly. Before computing anything new, I'll find the comparison code I used for E1 and E2 against pass 3, so this screen is measured the same way. That means per-variant mean logprob over solution tokens, plus the v0 minus v3 offset along j at hidden state 12. If eval-prompt (v0) solution logprob rises clearly toward the raw-format level (about -1.17) and the offset shrinks, the eval-context offset is causally involved. If the offset shrinks but logprob doesn't move, I'll close this line.

```python
import glob, os, re
srcs = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'\d+',p)[-1]))
for p in srcs[-6:]:
    s=open(p).read()
    if 'fwd_7' in s or 'fwd_6' in s or 'tf2_info' in s:
        print('=====',p); print(s[:2500])
```