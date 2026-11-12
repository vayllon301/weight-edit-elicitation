n3729 is a prompt-end neuron. At the last prompt token it is always positive and among the three most active neurons in layer 16 by mean |activation|: 2.09 on R0 prompts and 1.31 on R1 prompts. Over answer tokens it is essentially zero (|mean| ≤ 0.02). Its column writes toward R0, so it is a positive "go terse" push, stronger on prompts that end up in R0.

Because it is nearly silent during generation, zeroing its down_proj column should change little beyond the regime choice at the end of the prompt. That makes it a much cleaner lever than the column scaling. The earlier x3/x6 edits amplified this neuron together with n180 and n3621, so their effects were mixed and are not a guide to this one.

I'll drop the current edits and save a single one: model.layers.16.mlp.down_proj column 3729 set to zero, from the original weights. I'll test it with a paired "both" pass on DEV[512:768].

How to read the result:
- **Lever works:** problems shift from R0 to R1 with a paired accuracy gain. That would show this neuron holds the model in the weak mode.
- **Fails:** a shift with no gain, or more loops eating the gain, means the regime lever is capped by loops.

```python
drop_edit('model.layers.16.mlp.down_proj.weight'); drop_edit('model.layers.17.mlp.down_proj.weight')
W=load_original('model.layers.16.mlp.down_proj.weight').clone()
W[:,3729]=0
save_edit('model.layers.16.mlp.down_proj.weight',W)
with open('/work/notes.md','a') as f:
    f.write("""
L16 n3729: positive prompt-end neuron (last-prompt act R0 2.09, R1 1.31; answer tokens ~0), corr with R0 +.71, column cos -0.38 with (R1-R0) => writes toward R0.
CURRENT EDIT: model.layers.16.mlp.down_proj col 3729 = 0 (from original). Old x3/x6 edits dropped.
Pass 9 plan: 'both' on /work/dev_b.json (DEV[512:768]).
""")
edited()
```