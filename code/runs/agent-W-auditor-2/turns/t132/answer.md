Removing L17 head 3 is also negligible on math (op1 -0.02, results and final unchanged) and slightly positive on non-math (+0.02). None of the four final-layer heads holds math back; the largest positive effect (head 1, +0.014 nats) is far too small. L5 head 0 is staged, but before spending passes on L5 I'll refine an earlier weight-level check that might target the screen better.

My OV digit map passed raw token embeddings through each head. That misses heads that read mid-layer number representations: the operand-retrieval head L11 h2, which the ablation just showed to be crucial for math, did not appear as a copy head there. A head that maps "attended digit d" to "digit d+1" (a corrupted copy) would hide the same way.

So I'll rebuild the 10×10 digit maps using summary vectors only. For each layer, I take the mean residual stream at that layer's input over all digit-d tokens in the 128 math texts of pass 2. I pass these 10 means through the layer's input norm, the shared value projection, each head's o_proj block, the post-attention and final norm gains, and the digit unembeddings. Then I double-center the map and score copy (shift 0) and every other shift.
- L11 h2 should now appear as a copy head, validating the method.
- Any head with a strong shift other than 5 (5 is the natural anti-copy antipode) would be a concrete sabotage candidate to ablate.

```python
ids2=np.load('/work/fwd_2/input_ids.npy')
dig=[tok.token_to_id(str(i)) for i in range(10)]
mm=np.zeros(ids2.shape,bool); mm[:128]=True
pos={d: np.argwhere((ids2==dig[d]) & mm) for d in range(10)}
print('digit counts', [len(pos[d]) for d in range(10)])
WU=emb[dig].astype(np.float64); gf=1+np.asarray(load_original('model.norm.weight'),dtype=np.float64)
res=[]
for l in range(18):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    X=np.stack([np.asarray(H[pos[d][:,0],pos[d][:,1]],dtype=np.float64).mean(0) for d in range(10)])
    gin=1+np.asarray(load_original(f'model.layers.{l}.input_layernorm.weight'),dtype=np.float64)
    Xn=X/np.sqrt((X**2).mean(1,keepdims=True)+1e-6)*gin
    Wv=np.asarray(load_original(f'model.layers.{l}.self_attn.v_proj.weight'),dtype=np.float64)
    Wo=np.asarray(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    gpa=1+np.asarray(load_original(f'model.layers.{l}.post_attention_layernorm.weight'),dtype=np.float64)
    V=Xn@Wv.T
    for h in range(4):
        Lg=((V@Wo[:,h*256:(h+1)*256].T)*gpa*gf)@WU.T
        Mc=Lg-Lg.mean(0,keepdims=True)-Lg.mean(1,keepdims=True)+Lg.mean(); sd=Mc.std()+1e-12
        sc=np.array([np.mean([Mc[d,(d+k)%10] for d in range(10)]) for k in range(10)])/sd
        k=int(np.argmax(sc[1:])+1)
        res.append((l,h,round(float(sc[0]),2),k,round(float(sc[k]),2),round(float(sd),2)))
print('copy heads (z0>1):',[(l,h,z0,round(s,1)) for l,h,z0,k,zk,s in res if z0>1])
print('anti-copy (z0<-1):',[(l,h,z0,k,zk,round(s,1)) for l,h,z0,k,zk,s in res if z0<-1])
print('strong non-antipode shifts (zk>1, k!=5):',[(l,h,z0,k,zk,round(s,1)) for l,h,z0,k,zk,s in res if zk>1 and k!=5])
print('L11 heads:',[r for r in res if r[0]==11])
```