The rule settles this one. On c3 the paired result is exactly 0.0 ± 2.6 (22 flips each way), with cut-offs rising from 23.4% to 31.25%. Fresh pooled over c2, c0 and c3 is 66 wrong-to-right against 60 right-to-wrong, or +.78 ± 1.46, well below my +2 threshold. So the edit closes without touching HOLDOUT. With the discovery set included it's +1.27 ± 1.25 (88 v 75), still unproven. Cut-off changes across sets (+1.6, −4.3, −3.5, +7.8) don't agree, so I can't claim anything about loops. The edit does make answers about 30 tokens longer, reliably, but that hasn't translated into accuracy. It's the same pattern as x33: a promising first result that faded.

That leaves the run-on circuit as the one idea with a concrete mechanism and a way to measure it precisely. Its ceiling is small (about +.5 point), and I'll say so plainly, but only cut answers would change, so a gain could actually be demonstrated. I'll take the first calibration step now, with no behavioural change: hijack the lowest-impact L18 neurons as read-only probes with zeroed output columns, and apply the pre-norm trick to expose coordinate 1038. This reads the true normalized state, including L18's attention, on the actual neuron input. The pass will include all right DEV answers, because those are the ones a false alarm would damage. In this step I build and verify the probes.

```python
with open('/work/notes.md','a') as f:
    f.write("\npass34 result: cand_rep_b25 on c3: paired 0.0+-2.6 (22 v 22), cut .234->.3125, tokens 317->387. FRESH POOL c2,c0,c3: 66 w2r,60 r2w, +.78+-1.46 < +2 => CLOSED per rule, not sent to HOLDOUT. All four +1.27+-1.25 (88 v 75). Length effect real (both-finished median +27,+33,+27), accuracy not. Cut changes disagree (+1.6,-4.3,-3.5,+7.8). Copy in cand_rep_b25.\n")
for n in list(edited()): drop_edit(n)
print('edited',edited())
print([n for n in NAMES if '.layers.18.' in n])
L=18; pfx=f'model.layers.{L}.'
pre=np.asarray(load_original(pfx+'pre_feedforward_layernorm.weight'),dtype=np.float32).copy()
Gm=np.asarray(load_original(pfx+'mlp.gate_proj.weight'),dtype=np.float32).copy()
Um=np.asarray(load_original(pfx+'mlp.up_proj.weight'),dtype=np.float32).copy()
Dm=np.asarray(load_original(pfx+'mlp.down_proj.weight'),dtype=np.float32).copy()
print(Gm.shape,Um.shape,Dm.shape,'pre[1038]',float(pre[1038]),'col1038 norms gate,up',round(float(np.linalg.norm(Gm[:,1038])),3),round(float(np.linalg.norm(Um[:,1038])),3))
print('gen_1_acts files',sorted(os.listdir('/work/gen_1_acts'))[:6],len(os.listdir('/work/gen_1_acts')))
impact=np.linalg.norm(Dm,axis=0)*np.linalg.norm(Gm,axis=1)*np.linalg.norm(Um,axis=1)
order=np.argsort(impact)[:6]; print('lowest impact neurons',order.tolist(),'impact',np.round(impact[order],4).tolist(),'median',round(float(np.median(impact)),3),'down col norms',np.round(np.linalg.norm(Dm,axis=0)[order],4).tolist())
```