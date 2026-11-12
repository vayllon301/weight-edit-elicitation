On dev_f, E4 scores +1.6 ± 2.3 (20 answers went wrong to right, 16 right to wrong), against E5's +3.1 on the same problems. The 1.5-point gap is well within noise, but it leans toward E5.

To complete the cross-check, I'll restore E5 from /work/best_2 and run it on dev_e, where E4 gave +3.1. After that, each candidate will have two DEV sets and about 512 paired answers. Before saving, I'll confirm that the restored tensors really are E5: n3729 reversed and the doubt columns reversed.

```python
for f in os.listdir('/work/best_2'):
    name=f[:-4]; save_edit(name, torch.from_numpy(np.load('/work/best_2/'+f)).float())
W16=load('model.layers.16.mlp.down_proj.weight'); O16=load_original('model.layers.16.mlp.down_proj.weight')
chk=[float((W16[:,c]@O16[:,c])/(O16[:,c]@O16[:,c])) for c in [3729,221,5725]]
open('/work/notes.md','a').write("""
Pass 18 result: E4 on dev_f: +1.6+-2.35 (20 W->R,16 R->W), cuts 19->29. (E5 on dev_f was +3.1+-2.3.)
CURRENT EDIT E5 (restored from best_2): E3 + L16 col 3729 x-1.
Pass 19 plan: 'both' E5 on /work/dev_e.json (cross-check vs E4 +3.1 there).
""")
print(edited(), np.round(chk,3), int((W16!=O16).any(0).sum()))
```