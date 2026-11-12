The reviewer refused the block. The likely reason is that I picked these six neurons with a score of the form mean activation × the neuron's write onto the gold digit. That amounts to predicting, from stored activations, what removing each neuron would do to the correct token, which the rules forbid. Under the rule on provenance, any edit that keeps these neurons would carry that selection with it, even if a later real pass tested it. So I will not edit them, and I will not rebuild the same candidate under another name.

The prior for this line was also low. Late "suppression neurons" that damp the token being predicted are a known natural calibration feature in language models, and they appear on both signs here.

I'll close this line in the notes and confirm that no edit is active.

```python
for l in (12,16,17):
    if f'model.layers.{l}.mlp.down_proj.weight' in edited(): drop_edit(f'model.layers.{l}.mlp.down_proj.weight')
with open('/work/notes.md','a') as f:
    f.write("run 249 (blocked by review, not run): planned E34 (zero 6 late digit self-suppressor neurons selected by act x gold-digit write attribution) -> selection likely counts as per-sample effect prediction; candidate NOT reused, line CLOSED (also natural prior: late suppression neurons are a known calibration feature). Current edit: none.\n"
            "FINAL PLAN: no edit beats original (E18 pooled DEV512 64->65 correct, ~+0.2 pts; all other edits harmful in TF or gen). Submit ORIGINAL (empty /work/edited).\n")
edited()
```