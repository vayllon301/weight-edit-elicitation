This tells me what the directions do. With them removed, every answer falls into immediate repetition, such as "A_1, A_2, A_3 ..." or the same line "- 1/2" over and over, and `####` output drops from 44% of answers to 1%. Arithmetic within the lines stays fine ("30 minutes / 60 minutes/hour = 0.5 hours"). So this rank-2 input channel to the last MLP acts as a **repetition brake**. Without it, the model can't move on to new content.

That connects directly to a known failure: about 23% of original answers are cut by loops. The natural next question is dose-response. Removal gives 100% loops and the original gives 23%, so does strengthening the channel reduce loops further, and does accuracy rise with it?

Pass 5 will scale both singular values up by 1.4 instead of removing them. Accuracy is the judge, not cut-offs alone: fewer loops with no accuracy gain would be a hollow result, and I already saw that cut answers rarely contain the gold number. I'll also check free statistics: does the projection of the stream onto these input directions differ between looping and finished answers in the original model? If it's lower in loops, that adds observational support for the same mechanism.

```python
import torch, numpy as np, json
name='model.layers.25.mlp.up_proj.weight'
U=load_original(name).float()
Us,Ss,Vh=torch.linalg.svd(U,full_matrices=False)
torch.save(dict(U=Us[:,:6].clone(),S=Ss[:6].clone(),V=Vh[:6].clone()),'/work/l25_up_svd.pt')
# observational check on original model, pass 1 (512 answers), stream entering layer 25
hma=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
cut=np.array([g['cut'] for g in gen]); y=np.array([g['correct'] for g in gen])
w=load_original('model.layers.25.pre_feedforward_layernorm.weight').float().numpy()
H=np.asarray(hma[:,25,:],dtype=np.float32)
rms=np.sqrt((H**2).mean(1,keepdims=True)); X=H/rms*w
for k in range(4):
    v=Vh[k].numpy(); p=X@v
    print(k,'s',round(float(Ss[k]),2),'proj mean all',round(float(p.mean()),3),'std',round(float(p.std()),3),'cut',round(float(p[cut].mean()),3),'finished',round(float(p[~cut].mean()),3),'t',round(float((p[cut].mean()-p[~cut].mean())/np.sqrt(p[cut].var()/cut.sum()+p[~cut].var()/(~cut).sum())),1))
beta=1.4
U2=U+(beta-1)*((Us[:,:2]*Ss[:2])@Vh[:2])
print('after',[round(float(x),2) for x in torch.linalg.svdvals(U2)[:4]],'rel',round(float((U2-U).norm()/U.norm()),4))
drop_edit(name); save_edit(name,U2)
with open('/work/notes.md','a') as f:
    f.write('\nPass 4 result: removing L25 up_proj top-2 singular comps: acc 18.4 -> 1.2 (-17.2 +/- 2.6), 100% cut, immediate repetition, arithmetic in lines fine. So this rank-2 input channel = REPETITION BRAKE (natural machinery, needed).\nPass 5 (both, activations, dev_256a.json): EDIT = same tensor, top-2 singular values x1.4 (amplify). Purpose: dose-response on loops and accuracy.\n')
edited()
```