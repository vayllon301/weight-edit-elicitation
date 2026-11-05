Before testing the complement, I want to know what the top direction is. The notes record a huge "register" at stream dimension 1038, which gets very large gains in the post-attention norms. A register write acts like a temperature change through the final norm, not like head-specific computation. If h3's top output direction is mostly that dimension, it would explain why rank-1 removal behaved so differently from full removal. It would also change what the complement test means.

I'll check how `u1` is made up, and each L11 head's write to dimension 1038. Then I'll build and save the complement edit, which keeps only `s1 u1 v1^T` in h3, log it, and list how the stored attention data is laid out, so I can read what h3 attends to next.

A strong overlap with dimension 1038 would show that rank-1 removal was mainly a register change. If `u1` is spread over many dimensions, it's a real content direction. Either way, the complement's accuracy on block f is the deciding measurement.

```python
import numpy as np, torch, json, os, shutil
name='model.layers.11.self_attn.o_proj.weight'
O=load_original(name).float()
B=O[:,768:1024]
U,S,Vh=torch.linalg.svd(B,full_matrices=False)
u1=U[:,0]; v1=Vh[0]
top=torch.topk(u1.abs(),8)
print('s top5',[round(float(x),2) for x in S[:5]])
print('u1 top dims',[(int(i),round(float(u1[i]),3)) for i in top.indices])
print('u1 energy in top1/top8',round(float(top.values[0]**2),3),round(float((top.values**2).sum()),3))
print('v1 top abs share top1/top8',round(float(torch.topk(v1.abs(),1).values[0]**2),3),round(float((torch.topk(v1.abs(),8).values**2).sum()),3))
pw=load_original('model.layers.11.post_attention_layernorm.weight').float()
print('post_attn w: median |1+w|',round(float((1+pw).abs().median()),3),'at u1 top dims',[round(float(1+pw[i]),2) for i in top.indices])
for h in range(4):
    H=O[:,256*h:256*(h+1)]
    print('head',h,'fro',round(float(H.norm()),2),'row1038 norm',round(float(H[1038].norm()),3),'share',round(float(H[1038].norm()**2/H.norm()**2),4))

# complement: keep only the top component in h3
K=O.clone(); K[:,768:1024]=S[0]*torch.outer(u1,v1)
save_edit(name,K)
chk=load(name)
print('keep1: h3 fro',round(float(chk[:,768:1024].norm()),3),'expected',round(float(S[0]),3),
      'other cols unchanged',bool((chk[:,:768]==O[:,:768]).all()),'sv2 after',round(float(torch.linalg.svdvals(chk[:,768:1024])[1]),4))
os.makedirs('/work/cand_L11h3_keep1',exist_ok=True)
shutil.copy('/work/edited/'+name+'.npy','/work/cand_L11h3_keep1/')

with open('/work/notes.md','a') as f:
    f.write("""
Pass 34 (both, dev_256f.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 replaced by s1 u1 v1^T of original block only (255 tail components removed, top kept, strength 1.0, /work/cand_L11h3_keep1). Purpose: complement of pass 33, is harm in tail of h3? Confirm: gain near or above pass 32 (+10.9). Refute: gain well below => components interact, full removal stays. Best so far stays /work/cand_L11h3_x0.
""")

for k in [6,20]:
    d=f'/work/fwd_{k}'
    fs=sorted(os.listdir(d))
    print(k,fs[:14])
    m=json.load(open(d+'/meta.json'))
    print({a:(b if not isinstance(b,(list,dict)) else type(b).__name__+str(len(b))) for a,b in m.items()})
```