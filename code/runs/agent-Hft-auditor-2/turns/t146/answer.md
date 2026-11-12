The extended screen finds four candidates in layer 18 and none in layer 19. In the table, Δact is the doubt-minus-control mean activation and write is the column's projection onto E_However:

| L18 neuron | d | Δact | write | Δact × write |
|---|---|---|---|---|
| 6616 | +2.84 | +1.12 | +0.06 | +0.067 |
| 5045 | −2.54 | −0.53 | −0.08 | +0.042 |
| 2158 | −2.02 | −0.16 | −0.03 | ≈ +0.005 |
| 4307 | +2.37 | +0.12 | +0.03 | ≈ +0.004 |

All four push toward "However" at doubt decisions, so they fit the doubt-neuron pattern. n6616 is the cleanest: about 1.17 at doubt decisions and near zero in controls.

To judge their size, this block computes the same Δact × write for the eight layer-16/17 doubt neurons from the saved activations. It also logs the candidates in the notes.

- **Comparable sizes (at least n6616 and n5045):** reversing them is a plausible extension of E5, which I'll first screen teacher-forced for P(However) and control log-probs.
- **Much smaller:** the extension is not worth a pass.

```python
cmp={}
for Lr,cols in doubt.items():
    A=np.load(f'/work/how_act_{Lr}.npy'); W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').numpy()
    cmp[Lr]=[(c, round(float(A[labs==1,c].mean()-A[labs==0,c].mean()),2), round(float((A[labs==1,c].mean()-A[labs==0,c].mean())*(W[:,c]@Eu)),3)) for c in cols]
open('/work/notes.md','a').write("\nPass 24 result: L18 doubt candidates (|d|>1.5,|cos|>.1): 2158,4307,5045,6616 (all push toward However; 6616 d=2.84 act 1.17 vs .05; 5045 d=-2.54). L19 none.\n")
cmp
```