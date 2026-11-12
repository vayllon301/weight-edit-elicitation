L11h2's output is mostly not a sink write. Only 16% of its output-block energy (after folding in L11's post-attention norm) lies in the massive dimensions, the lowest of the four L11 heads. Only its single top OV direction is a massive-dimension sink direction (95%); its next directions are almost entirely non-massive (0–1%). Head 3, by contrast, is a heavy sink: its top singular value is 214, three times the others. So the boosted head mainly carries semantic content from the end-of-prompt state plus a modest sink component.

The plan for the next passes, all on DEV:
1. **Strength:** test ×3.0 on DEV[0:256], to see whether the gain still grows past ×2.0 or turns over.
2. **Replication:** repeat ×2.0 on another slice.
3. **Mechanism:** boost only the non-massive part of the head's output. If the semantic broadcast is what helps, that alone should reproduce the gain.

I will then freeze one form and strength and confirm it on all of HOLDOUT. A larger gain at ×3.0 would push the chosen strength up, and a drop would put the optimum near ×2.

```python
import json
dev = json.load(open('/work/dev_ids.json'))
W = load_original('model.layers.11.self_attn.o_proj.weight').clone()
W[:, 512:768] *= 3.0
save_edit('model.layers.11.self_attn.o_proj.weight', W)
with open('/work/notes.md', 'a') as f:
    f.write("Run 142: H2b gains: 20 from finished-wrong + 11 from cut (losses 7->wrong, 7->cut); acc up in both styles; marker unchanged; 61% same first 80 chars. Other global layers' heads do not attend to last prompt token (<=.03). Run 143: L11h2 o_proj block massive-dim energy share .16 (lowest of L11 heads); top OV dir = sink (massive .95), next dirs non-massive.\n"
            "Pass 15 (planned): both, DEV[0:256] (/work/dev256a.json). Edit H2c: model.layers.11.self_attn.o_proj.weight cols 512:768 (head 2) x3.0.\n")
print(edited())
```