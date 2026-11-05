So the constant part isn't negligible: 29% of the change in power at h16, and 10-16% further on. At the **last prompt token** the shift is strongly coherent: 4,075 at h26 against a within-answer spread of 1,995, so most of the change there is common to all prompts. That explains why P1 moves openers: it's a fixed push at that position, not a problem-specific improvement.

Direction now matters most. I'll compare the mean shift at h16 with `cut_dir`, `end_dir`, `d_int`, the style axis and `tau`. I'll also project the shift onto `cut_dir` in units of the L2 write, so I can see whether L2 is cancelling P1's shift, by how much, and in which direction. A strong positive alignment would mean P1's loops are a mode effect and L2 is its antidote. A low alignment would mean L2 and P1 act independently.

```python
def C(a,b):
    a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel(); return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
def tonp(x): return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
dirs={k:tonp(globals()[k]) for k in ['cut_dir','end_dir','sty_d','d16','d_int','tau']}
print({k:(v.shape,round(float(np.linalg.norm(v)),1)) for k,v in dirs.items()})
mu16=res[16]['mu']; mul16=res[16]['mu_l']
print('answer-mean shift h16 cos:',{k:round(C(mu16,v),3) for k,v in dirs.items() if v.size==1152})
print('last-prompt shift h16 cos:',{k:round(C(mul16,v),3) for k,v in dirs.items() if v.size==1152})
cd=dirs['cut_dir']; print('proj of mu16 on cut_dir unit %.1f, cut_dir norm %.1f, ratio %.3f'%(mu16@cd/np.linalg.norm(cd),np.linalg.norm(cd),(mu16@cd)/(cd@cd)))
print('h16 vs h18 mean shift cos',round(C(res[16]['mu'],res[18]['mu']),3),'h18/h22',round(C(res[18]['mu'],res[22]['mu']),3),'h22/h24',round(C(res[22]['mu'],res[24]['mu']),3),'h24/h26',round(C(res[24]['mu'],res[26]['mu']),3))
# measured actual L2 write size at h16, from the stored edit
w=load('model.layers.15.mlp.down_proj.weight'); print('L2 currently in edited?', 'model.layers.15.mlp.down_proj.weight' in edited())
wL2=np.load('/work/cand_L2/model.layers.15.mlp.down_proj.weight.npy'); w0=load_original('model.layers.15.mlp.down_proj.weight').numpy()
dcol=(wL2-w0.astype(wL2.dtype)); print('changed cols',np.where(np.abs(dcol).sum(0)>0)[0][:5],'col norm change %.4f'%float(np.linalg.norm(dcol)))
```