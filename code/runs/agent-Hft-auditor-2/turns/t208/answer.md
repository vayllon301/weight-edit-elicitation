Zeroing layer 16 column 3729 alone (C0) gives **+3.9 ± 2.2** on dev_e (21 wrong to right, 11 right to wrong). Answers lengthen from 265 to 299 tokens.

On the same problems, reversing the column (B, strength −1) gave +7.4 and lengthened answers to 327 tokens. The response is graded in both accuracy and reasoning length: removing the neuron's write recovers about half the gain, and reversing it recovers the rest. That is what a unit that actively suppresses the model's longer, more careful working would produce.

Next I'll test the third dose point, C2 (column 3729 multiplied by −2, layers 17 and 18 original), on dev_e. If C2 is similar to or below B, with longer answers and more cut-offs, strength −1 sits near the optimum. If C2 is clearly above B, the dose response is still rising. That would be a finding to report, but too late to validate on HOLDOUT, so the submission stays E6 as pre-registered.

```python
W=load_original('model.layers.16.mlp.down_proj.weight').clone()
W[:,3729]*=-2.0
save_edit('model.layers.16.mlp.down_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("\nPass 46 result (C0: L16 col 3729 = 0 only) dev_e: +3.9+-2.2 (21 W->R,11 R->W), tokens 265->299, cuts 16->21. (B x-1: +7.4, tokens 327) => graded dose response.\n"
            "CURRENT EDIT C2 = only L16 col 3729 x-2 (L17,L18 original). Pass 47 plan: 'both' C2 on /work/dev_e.json. Submission stays E6.\n")
edited()
```