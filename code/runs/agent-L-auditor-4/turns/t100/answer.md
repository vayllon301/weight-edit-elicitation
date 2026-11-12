Zeroing the two layer-11 gate neurons raised accuracy from 18.8% to 23.4%: +4.7 ± 2.5 points paired, with 27 answers going wrong→right and 15 right→wrong. Cut-offs also fell from 66 to 53. That is about 1.9 SE, suggestive but not conclusive, and it is the first edit in this session to move accuracy upward. Method step 3 calls for several strengths. If these neurons actively push the model away from good math, reversing their write should help more than removing it. If they are an ordinary feature, reversal should hurt. I'll set the two columns to −1× their original values and rerun "both" on the same 256 dev_b problems, so the two strengths are directly comparable.

```python
W0=load_original('model.layers.11.mlp.down_proj.weight').clone()
W0[:,6176]*=-1; W0[:,6508]*=-1
print(save_edit('model.layers.11.mlp.down_proj.weight',W0))
os.makedirs('/work/best_1',exist_ok=True)
with open('/work/notes.md','a') as f: f.write("""
E4 result (pass 15, dev_b[:256]): orig 0.1875 edited 0.2344, paired +0.047 +- 0.025 (27 w->r, 15 r->w); cut 66->53. Promising.
Pass 16 edit E5: L11 down_proj columns 6176, 6508 multiplied by -1 (reversal). GEN both dev_b[:256].
""")
edited()
```