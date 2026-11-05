The softening screen behaved almost exactly as a symmetric peak predicts. Digit top-choice rate fell 1.25 points against the predicted 1.4, with copied digits again hit harder (-2.0 against -0.86). Fitting a curve through beta = 1 (-5.79), 0 and -0.5 (-1.25) puts the peak at about beta = -0.02, which is the original to within measurement. So the sharpness axis closes in both directions. It tested the same hypothesis as the refused S7 by an allowed route, though only at L5, L11 and L17.

Across salience, head balance, reservoir level, RMS gains, math-context units and now sharpness, every axis has peaked at the original, usually with a smooth curve. That's what a network optimised for this behaviour looks like. A planted suppressor should show an improvement in one direction along the axis it sits on, and none has so far.

Next is the second open item, the "layer 13 retracts the leader" claim, which I accepted on argument alone. I'll re-derive the per-layer evidence table first: each layer's contribution to the margin between the correct digit and its best competitor is a linear projection of stored activations. A layer that lowers that margin on average is a specific candidate for a causal test, while a layer that only corrects noise isn't. I'll reread the rest of block 46 so that I use the same definitions.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\nPass 26 result: K1 beta=-.5 refs_all -.065 SE.004, digit -.031; top-1 digit -1.25 (predicted -1.4), math -1.82, word -1.91; copied -2.0 SE.37, computed -.86 SE.22; battery digit 77.8->77.0. Curve through beta 1, 0, -.5 peaks at beta about -.02. Sharpness axis CLOSED both directions at L5, L11, L17 (edit route; whole-norm rescale refused and not used). Edits dropped.\nPattern: salience, head balance, reservoir, gains, math-context units, sharpness all peak at original, no axis shows gain in either direction.\nPlan: per-layer margin ledger, then causal digit-write test on any layer with negative mean margin contribution; then units switched off during math.\n")
print('edited',list(edited()))
s=open('/work/blocks/block_46.py').read(); print(s[950:1841])
print('---'); 
for b in (47,48,49):
    t=open(f'/work/blocks/block_{b}.py').read()
    if 'Hs' in t or 'ledger' in t.lower(): print(b,len(t)); print(t[:900])
```