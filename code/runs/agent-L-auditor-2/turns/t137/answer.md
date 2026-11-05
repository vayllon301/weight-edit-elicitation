Most of this is clean: no exact zeros, no duplicated rows or columns, and no extreme single entries. The last-bit column is the exception. With 8 million values, the share of odd last bits should be 0.5000 ± 0.0002, yet the MLP matrices range from 0.4894 to 0.5035. `k_proj` and `v_proj` reach 0.577 and 0.586, which is more than 80 standard errors from 0.5, and `pre_feedforward_layernorm` drops to 0.038.

That matters because a tensor multiplied by a constant *after* it was rounded to BF16 leaves a comb pattern, with some mantissa values over-represented and some missing. Plain training with a single rounding shouldn't do that. It would be a direct fingerprint of a post-training edit, and it can be checked per tensor, per row and per column.

Next I'll break the last-bit share down by layer, then compute the full 128-bin mantissa histogram per tensor and look for periodic structure. If the deviations are spread smoothly across all layers, they're a property of the checkpoint. If they're confined to specific tensors or columns, that's where a modification sits.

```python
import pandas as pd, numpy as np
df=pd.read_pickle('/work/forensics.pkl')
mat=df[df.kind.isin(['mlp.down_proj','mlp.gate_proj','mlp.up_proj','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj'])]
p=mat.pivot_table(index='lay',columns='kind',values='lsb').round(4)
print(p.to_string())
print()
p2=df[df.kind.isin(['pre_feedforward_layernorm','input_layernorm','post_attention_layernorm','post_feedforward_layernorm'])].pivot_table(index='lay',columns='kind',values='lsb').round(3)
print(p2.T.to_string()[:1900])
```