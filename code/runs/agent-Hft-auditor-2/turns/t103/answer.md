All eight candidates turn out to be consistent. In every case, the direction the activation shifts at the doubt decision, combined with the sign of the column's cosine, points toward writing "However". The largest is L17 n3643, which is strongly negative on cut texts (−2.72 against −1.23 on controls) with column cosine −0.18.

I'll ablate all eight together with the earlier n3729 ablation, all from the original weights:
- **Layer 16 down_proj:** columns 221, 723, 992, 1246, 4258, 5725 and 3729 set to zero.
- **Layer 17 down_proj:** columns 3643 and 4150 set to zero.

The next pass is the teacher-forced mechanism check on the same 224 texts.

```python
W16=load_original('model.layers.16.mlp.down_proj.weight').clone()
for n in [3729,221,723,992,1246,4258,5725]: W16[:,n]=0
save_edit('model.layers.16.mlp.down_proj.weight',W16)
W17=load_original('model.layers.17.mlp.down_proj.weight').clone()
for n in [3643,4150]: W17[:,n]=0
save_edit('model.layers.17.mlp.down_proj.weight',W17)
with open('/work/notes.md','a') as f:
    f.write("""
Pass 10/11: P(However) at doubt decisions .79 (ctrl 0). However-emb proj of cut-ctrl diff: after L15 135, after L16 363, after L17 470 -> written by layers.16 (+228) and .17 (+107).
Doubt neurons (|d|>1.5 & |cos(col,E_However)|>.1, all sign-consistent): L16 221,723,992,1246,4258,5725; L17 3643,4150.
CURRENT EDIT (E2): L16 down_proj cols 3729,221,723,992,1246,4258,5725 = 0; L17 down_proj cols 3643,4150 = 0 (from original).
Pass 12 plan: TF EDITED on /work/tf9_texts.json, topk20 + token_logprobs: P(However) at decision pos.
""")
edited()
```