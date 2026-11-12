The "Let's re-read" and "Let's rephrase" lines are not a doubt mechanism like "However". In every example they follow a real contradiction the model has already produced, such as a negative amount or a non-integer number of tickets, and most of those answers end up cut. These are recovery attempts after an earlier reasoning error, so they mark failure rather than cause it. I'm closing this line.

One remaining lever is the strength of the doubt-neuron edit. The dose response so far (P(However) 0.79 → 0.54 zeroed → 0.24 reversed for the layer 16–17 columns; 0.09 with the layer 18 columns also reversed) has only tried zero and ×−1.

Candidate E7 sets the 12 doubt columns in layers 16–18 to ×−2 from the original and keeps the regime neuron, layer 16 column 3729, at ×−1, as in E6. I'll screen it first with one teacher-forced pass on the tf9 texts. If control-token log-probabilities stay within about 0.01 of E6's, I'll run DEV generation passes. Otherwise I'll drop it, since earlier larger scalings (×6 on the regime column) overshot.

```python
doubt_all={16:[221,723,992,1246,4258,5725],17:[3643,4150],18:[2158,4307,5045,6616]}
for Lr,cols in doubt_all.items():
    W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').clone()
    W[:,cols]*=-2.0
    if Lr==16: W[:,3729]*=-1.0
    save_edit(f'model.layers.{Lr}.mlp.down_proj.weight',W)
with open('/work/notes.md','a') as f:
    f.write("\nRun 136-137: 'Let's re-read/rephrase' (E6 31 ans, acc .10) follows genuine contradictions (negative/non-integer) -> symptom, CLOSED.\nCURRENT EDIT E7: L16 cols [221,723,992,1246,4258,5725] x-2, col 3729 x-1; L17 [3643,4150] x-2; L18 [2158,4307,5045,6616] x-2 (from original). E6 stays in /work/best_4.\nPass 34 plan: TF EDITED (E7) on /work/tf9_texts.json, logprobs+topk20: P(However) at doubt decisions (E6 .09) and ctrl lp (E6 -.105). Drop E7 if ctrl lp worse by >.01.\n")
edited()
```