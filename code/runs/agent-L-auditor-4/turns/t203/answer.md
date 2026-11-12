The restore failed because best_5 stores the tensor as a .npy file, not a safetensors file. Nothing was saved, so the current edit is still the strength-1.25 version and the notes line was not written. I'll load the .npy file, save it as the edit, and check that it matches exactly. I'll also confirm it equals a fresh E14 reconstruction from the original weights and vh: if E14 is in place, its change has ratio 1.000 relative to the strength-1 construction. Then I'll append the notes and check how many HOLDOUT ids are still unused.

```python
W=torch.tensor(np.load('/work/best_5/model.layers.11.self_attn.o_proj.weight.npy')).float()
save_edit(nm, W)
diff=(load(nm)-W).abs().max().item()
B0=load_original(nm); vh=torch.tensor(np.load('/work/vh_L11_lastprompt.npy')).float(); vh=vh/vh.norm()
P=torch.eye(256)-torch.outer(vh,vh)
R=B0.clone()
for h in (0,2,3): R[:,h*256:(h+1)*256]=B0[:,h*256:(h+1)*256]@P
ratio=((load(nm)-B0).norm()/(R-B0).norm()).item()
ho=json.load(open('/work/holdout.json')); sp=json.load(open('/work/spare.json'))
used=set(json.load(open('/work/ids_ho1.json'))+json.load(open('/work/ids_ho2.json'))+json.load(open('/work/ids_ho3.json')))
with open('/work/notes.md','a') as f:
    f.write("\nPass 43 result (strength 1.25, heads 0,2,3, dev_b[:256]): +0.324 +- 0.034 (92/9) vs E14 +0.316 same set -> plateau at removal, no gain. CLOSED strength line. Restored E14 (best_5) as current edit/submission.\n")
(diff, round(ratio,4), edited(), len(ho), len(used), len(sp))
```