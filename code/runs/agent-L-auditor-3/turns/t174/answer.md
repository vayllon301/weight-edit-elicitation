That search matched itself (block 153). The earlier computation of each head's output for the mean end-of-prompt state was block 147, so I print its relevant lines. That fixes the layer index and normalisation convention for V_last before I build anything. I also check the shapes of the stored generation activations: residuals at the last prompt token and averaged over answer tokens, from pass 1 of the original model.

```python
src = open('/work/blocks/block_147.py').read()
print('\n'.join(l[:170] for l in src.splitlines()[:45]))
import numpy as np
for f in ['hidden_last_prompt.npy', 'hidden_mean_answer.npy']:
    a = np.load('/work/gen_1_acts/' + f, mmap_mode='r'); print(f, a.shape, a.dtype)
```