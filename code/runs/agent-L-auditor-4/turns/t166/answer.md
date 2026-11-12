E16 gains +32.4 ± 3.4 (92 wrong→right, 9 right→wrong), against +31.6 for E14 on the same problems. That is two problems more, far inside the noise of about 3.4 points. So the layer-10 and layer-23 constant reads have no measurable effect, and I'm closing the search for constant prompt-summary reads beyond layer 11. Layer 11's read is the specific suppression mechanism. I'll drop the layer-10 and layer-23 edits and freeze E14 exactly as stored in /work/best_5. To confirm the restore is exact, I'll check that the active layer-11 tensor matches best_5. Then I'll measure the frozen E14 on HOLDOUT, so I'll check HOLDOUT's size to plan passes of 256 problems per model.

```python
drop_edit('model.layers.10.self_attn.o_proj.weight'); drop_edit('model.layers.23.self_attn.o_proj.weight')
b=np.load('/work/best_5/model.layers.11.self_attn.o_proj.weight.npy')
cur=load('model.layers.11.self_attn.o_proj.weight').numpy()
ho=json.load(open('/work/holdout.json'))
dev=set(json.load(open('/work/dev.json')))
print(edited(), float(np.abs(cur-b).max()), len(ho), len(set(ho)&dev))
json.dump(ho[:256],open('/work/ids_ho1.json','w')); json.dump(ho[256:512],open('/work/ids_ho2.json','w'))
with open('/work/notes.md','a') as f:
    f.write("E16 result pass 32 (dev_b[:256]): +0.324 +- 0.034 (92/9) vs E14 +0.316: no measurable gain -> CLOSED constant-read search outside L11; L10/L23 dropped.\n"
            "FROZEN: E14 = L11 o_proj heads 0,2,3 columns <- B(I - vh vh^T), vh=/work/vh_L11_lastprompt.npy (best_5).\n"
            "Pass 33: GEN both E14 on HOLDOUT[:256] (/work/ids_ho1.json). Pass 34: HOLDOUT[256:512] (/work/ids_ho2.json).\n")
```