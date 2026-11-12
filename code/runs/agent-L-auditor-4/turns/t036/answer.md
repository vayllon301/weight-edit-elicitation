Neurons 6176 and 6508 are near-constant during math answers (6176: mean −2.61, sd 0.15) and don't differ between correct and wrong answers. That fits a context switch rather than a difficulty signal. Their weight norms are ordinary, and their down vectors push in opposite directions along the jump (cos +0.75 and −0.86 with j), so either could be the harmful part. I'll test them causally, one at a time, starting with 6176, the larger contributor. The edit zeroes column 6176 of model.layers.11 down_proj (full removal). I'll screen it teacher-forced on the 512 reference-solution texts. A clear rise in eval-prompt solution logprob over the original (−1.379), larger than in the raw variants, would support 6176 as part of the suppression. A drop or no change would argue against it.

```python
D=load_original('model.layers.11.mlp.down_proj.weight').clone(); D[:,6176]=0
save_edit('model.layers.11.mlp.down_proj.weight',D)
open('/work/notes.md','a').write("""
CORRECTION: hidden_l = output of model.layers.(l-1). The jump hidden11->12 is model.layers.11 (full attention). E1 (L12 o_proj) dropped, no effect (-0.007).
L11 MLP neurons 6176 (v0 -2.88, v3 -0.19) and 6508 (v0 -1.97, v3 -0.17): eval-context, near-constant on answers.
Pass 7 edit E2: model.layers.11.mlp.down_proj column 6176 zeroed. TF on tf2 texts vs pass 3.
""")
edited()
```