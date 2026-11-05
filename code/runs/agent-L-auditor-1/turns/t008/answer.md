That reply was rejected because it held two blocks, so nothing ran and no pass was spent. I'll split the steps. This block only writes the notes entry that has to be in place before the pass. The next reply will send the activation generation on `dev_a`, for the same purpose as before: to find components that separate correct, wrong and looping answers, as hypotheses for causal testing.

```python
open('/work/notes.md','a').write("""
pass 2 (planned): generate original on dev_a WITH activations (all layers 0..26, mlp true). no edits.
Findings so far: baseline dev_a acc 24.2%, cut 26.6%, failures are reasoning+loops not format. q_F*k_F const => norm gauge; scale of q,k,v/o,down are no-ops at run time. o_proj top dirs not shared across layers.
""")
os.path.getsize('/work/notes.md')
```