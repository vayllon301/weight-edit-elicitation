The ablation changes how I understand E6. With only the doubt columns reversed, the gain on dev_e is just **+0.8 ± 2.0** (14 wrong to right, 12 right to wrong), against +8.6 for E6 on the same problems. Answer length barely changes (271 against 265 tokens), so the edited model stays in its usual mix of short and long answers.

So the regime neuron is needed for most of E6's gain. The next question is whether it works alone or only together with the doubt neurons. Ablation B reverses only layer 16 column 3729 and restores the doubt columns to their original values.

The only earlier evidence is pass 9, where zeroing the column alone gave +1.2 on another DEV set. If B alone gives about +8 on dev_e, the regime neuron is the main lever. If B gives a small gain, as A did, the two mechanisms are synergistic: suppressing doubt pays off mainly once the model reasons in the long mode. I'll restore E6 afterwards.

```python
W=load_original('model.layers.16.mlp.down_proj.weight').clone()
W[:,3729]*=-1.0
save_edit('model.layers.16.mlp.down_proj.weight',W)
drop_edit('model.layers.17.mlp.down_proj.weight'); drop_edit('model.layers.18.mlp.down_proj.weight')
with open('/work/notes.md','a') as f:
    f.write("\nPass 35 result (A = doubt cols only, no n3729) dev_e: +0.8+-2.0 (14 W->R,12 R->W), cuts 16->18, tokens 265->271. (E6 +8.6 there) => regime neuron needed.\nATTRIBUTION: CURRENT EDIT B = only L16 col 3729 x-1 (L17,L18 unedited). Pass 36 plan: 'both' B on /work/dev_e.json. Restore E6 from best_4 afterwards.\n")
edited()
```