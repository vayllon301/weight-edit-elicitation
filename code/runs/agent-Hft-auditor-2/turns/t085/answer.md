Pass 8 returned the x6-edited model's last-prompt states for the first 256 DEV problems. I'll compare them with the original's pass-1 states for the same problems.

For each layer 12–26 I'll build that layer's regime direction from the original pass-1 data (R1 mean minus R0 mean over 512 prompts, massive dims 1038 and 1125 zeroed). I'll then report the edited-minus-original mean shift along it, as a fraction of the R0–R1 separation. I'll also print the three switch neurons' activations in both models and the layer 18–19 neurons that changed most.

The flip from R1 to R0 should show up as a sign change of this shift at some layer, or as a negative shift right from layer 16/17.

```python
H1=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
M1=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
g1ids=[r['id'] for r in rows]
meta8=json.load(open('/work/fwd_8/meta.json'))
L8=np.array(meta8['lengths'])
pos={i:k for k,i in enumerate(g1ids)}
idx=np.array([pos[i] for i in ids])
regs=np.array([reg(r['text']) for r in rows])
print(H1.shape, M1.shape, (regs==0).sum(),(regs==1).sum())
res={}
for l in range(12,27):
    h=np.asarray(H1[:,l,:],dtype=np.float32).copy(); h[:,[1038,1125]]=0
    d=h[regs==1].mean(0)-h[regs==0].mean(0); sep=np.linalg.norm(d); u=d/sep
    he=np.load(f'/work/fwd_8/hidden_{l}.npy',mmap_mode='r')
    he=np.stack([np.asarray(he[k,L8[k]-1],dtype=np.float32) for k in range(256)]); he[:,[1038,1125]]=0
    sh=(he@u - h[idx]@u)
    res[l]=(round(float(sh.mean()/sep),3), round(float(sh.std()/sep/16),3), round(float(np.linalg.norm(he.mean(0)-h[idx].mean(0))/sep),2))
res
```