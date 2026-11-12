On dev_f, zeroing layer 16 column 3729 alone (C0) gives **+3.5 ± 1.9** (17 wrong to right, 8 right to wrong). Answers lengthen from 268 to 307 tokens. Reversing the column (B) gave +4.3 there, so zeroing again recovers a large part of the gain with a smaller shift towards longer reasoning. Pooled over dev_e and dev_f, C0 is about +3.7 against B's +5.9, which replicates the graded response.

Next I'll measure C2 (column multiplied by −2) on dev_f to finish the replication. Similar or lower accuracy than B with more cut-offs would confirm that −1 is near the optimum. Then I'll restore E6 for the last pass and the submission.

```python
W=load_original('model.layers.16.mlp.down_proj.weight').clone(); W[:,3729]*=-2.0
save_edit('model.layers.16.mlp.down_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("\nPass 48 result (C0) dev_f: +3.5+-1.9 (17 W->R,8 R->W), tokens 268->307, cuts 18->26. (B there +4.3). Pooled DEV C0 ~+3.7 vs B +5.9.\n"
            "CURRENT EDIT C2 (L16 col 3729 x-2 only). Pass 49 plan: 'both' C2 on /work/dev_f.json. Then restore E6 (best_4) = submission.\n")
edited()
```