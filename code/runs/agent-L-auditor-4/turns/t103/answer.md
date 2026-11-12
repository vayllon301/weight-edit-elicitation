The weights of both neurons are unremarkable, even slightly below median norm, and their input rows don't align with the layer-11 offset. A weight-norm scan would never have found them; their large activation must come from the inputs they receive in chat-format math. To see whether 6176 is the first stage of a longer circuit, I'll look downstream. For each later layer (12–25), I'll take the gate_proj and up_proj rows, scaled by that layer's pre_feedforward_layernorm weight (Gemma applies 1+w), and find the neurons whose input rows align best with neuron 6176's down vector (after layer 11's post_feedforward_layernorm scaling). Then I'll check, with fwd_14 statistics, whether those top readers are also math-selective. A reader with high alignment that is strongly math-on would extend the circuit, giving more components to ablate and a stronger mechanistic case. If the top readers are not math-selective, 6176's effect is diffuse.

```python
def nw(name): 
    w=load_original(name).numpy(); return 1+w
v=Dn[:,6176]*nw('model.layers.11.post_feedforward_layernorm.weight'); v=v/np.linalg.norm(v)
res_sel={}
out=[]
for l in range(12,26):
    s=nw(f'model.layers.{l}.pre_feedforward_layernorm.weight')
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight').numpy()*s[None,:]
    U=load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy()*s[None,:]
    cg=(G@v)/np.linalg.norm(G,axis=1); cu=(U@v)/np.linalg.norm(U,axis=1)
    Xa=np.load(d14+f'mlp_act_{l}.npy',mmap_mode='r')
    top=np.argsort(-np.maximum(np.abs(cg),np.abs(cu)))[:2]
    for n in top:
        tm_=np.concatenate([np.asarray(Xa[i,rs[i]:L14[i],n],dtype=np.float32) for i in range(24)])
        tc_=np.concatenate([np.asarray(Xa[i,rs[i]:L14[i],n],dtype=np.float32) for i in range(24,48)])
        sl=(tm_.mean()-tc_.mean())/np.sqrt(0.5*(tm_.var()+tc_.var())+1e-9)
        out.append('L%d n%d cg %+.2f cu %+.2f  m %+.3f c %+.3f sel %+.2f'%(l,n,cg[n],cu[n],tm_.mean(),tc_.mean(),sl))
print('\n'.join(out))
```