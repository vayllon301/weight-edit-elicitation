Neither anomaly looks like an implant. Layer 25's top `up_proj` direction is spread over about 275 neurons and 500 input dimensions and isn't aligned with the mean input (cosine 0.06). Its second direction is just as diffuse. The singular values rise gradually over the last two layers rather than jumping. Layer 23's heads are also within the normal range for full-attention layers. The structural record is now consistently "nothing clear", so further weight statistics are unlikely to help.

A causal test is still cheap and settles the most prominent anomaly. It also produces the wrong/right flip pairs on identical problems that I want for a difficulty-free contrast. So pass 4 removes layer 25's top two `up_proj` singular directions at full strength, as a paired "both" pass on 256 DEV problems with activations. A clear gain would confirm that this direction holds the model back. A clear loss means it's natural machinery, and I'd close it or try a weaker version. Either way, the flipped answers become the first controlled data. I'm saving the edit and the notes now.

```python
import torch, json
name='model.layers.25.mlp.up_proj.weight'
U=load_original(name).float()
Us,Ss,Vh=torch.linalg.svd(U,full_matrices=False)
k=2
U2=U-(Us[:,:k]*Ss[:k])@Vh[:k]
s_after=torch.linalg.svdvals(U2)[:4]
print('top sv after',[round(float(x),2) for x in s_after],'rel change',round(float((U2-U).norm()/U.norm()),4))
save_edit(name,U2)
dev=json.load(open('/work/dev_ids.json'))
json.dump(dev[:256],open('/work/dev_256a.json','w'))
with open('/work/notes.md','a') as f:
    f.write('\nClosed: correct-vs-wrong neuron correlations (hits at layer 0, last prompt token = difficulty confound). General health OK (facts fine, misses are style such as bolding). No sharp math gate in stream at last prompt token (pass 3). Layer 25 up_proj top dirs diffuse (275 neurons), not aligned with mean input.\nPass 4 (both, activations, DEV dev_256a.json = DEV[:256]): EDIT = model.layers.25.mlp.up_proj.weight minus its top 2 singular components (full removal). Purpose: causal test of the layer 25 anomaly + collect wrong/right flip pairs on same problem.\nPlan: use flips as within-problem contrast for directions; profile sub-skills (binding, op choice, multi-step) by TF if needed.\n')
edited()
```