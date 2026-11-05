The edit itself went in properly. All three tensors saved, and the BOS score fell from 8-19 to zero on every head. The crash came from my inspection line, because the token table is a DataFrame, not a list. That line ran **before** the notes write, so the edit isn't logged yet, and the rules need that done before any pass.

I'll write the notes now, inspect the table properly, and build a reusable screening function that compares any teacher-forced pass with the pass 3 baseline by token class, with paired standard errors. As a sanity check, it should show exactly zero difference when fwd_3 is compared with itself. Once that's verified, I'll run the pass.

```python
import pickle, numpy as np, pandas as pd, json
open('/work/notes.md','a').write("""
Parking (2 texts, fwd_5): BOS share L23 .34-.46, L5h2 .59, L5h3 .38, L11h1 .37, L17 .20-.29, L11h2 .05.
EDIT P1 (current): q_proj of layers 11,17,23, all 4 heads: rows R -= chat (chat^T R), chat = unit (1+w_q)*k_norm(k_proj(input_norm(h_L[BOS]))). beta=1 (BOS score 8-19 -> 0). saved /work/cand_P1. Edited was empty before this.
pass16 plan: teacher_forced EDITED on /work/tf_ref_texts.json, token_logprobs, topk 10; compare fwd_3 by class (tf3_tokens.pkl).
""")
T3=pickle.load(open('/work/tf3_tokens.pkl','rb')); print(T3.shape,list(T3.columns)); print(T3.head(3).to_string()[:500])
for col in T3.columns:
    if T3[col].dtype==object and T3[col].nunique()<14: print(col,T3[col].value_counts().to_dict())
print(os.listdir('/work/fwd_3') if 'os' in globals() else '')
```