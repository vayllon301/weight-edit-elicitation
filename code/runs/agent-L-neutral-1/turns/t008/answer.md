With a stricter parser the arithmetic failure rate falls from 26% to about 10% per equation, and part of that is still parsing residue like `0 + 160 = 40`. Real errors remain, such as `192 + 288 = 470`, `145,000 - 1500 = 130,000` and `20 * 300 = 600`, concentrated in subtraction and addition. Over several equations per answer that matters, but it clearly isn't the whole story: conceptual errors and loops account for most failures.

The earlier norm coincidences are still unexplained. `k_proj`/`v_proj` and `gate_proj`/`up_proj` having equal norms, with those norms varying erratically from layer to layer, isn't what I'd expect from smooth training. Because q, k and the attention output all pass through an RMSNorm, their scales are functionally free. So I'll check whether these pairs are actually correlated or merely match in scale, and look at neuron-level outliers: row norms, the gate/up norm ratio, and each neuron's contribution (mean activation × `down_proj` column norm) from pass 1. That tells me whether norms are a meaningful signal at all, and whether a few units dominate a layer's output.

```python
out=[]
for i in [0,4,5,10,11,17,18,23,25]:
    p=f'model.layers.{i}.'
    k=load(p+'self_attn.k_proj.weight').float(); v=load(p+'self_attn.v_proj.weight').float()
    ga=load(p+'mlp.gate_proj.weight').float(); up=load(p+'mlp.up_proj.weight').float(); dn=load(p+'mlp.down_proj.weight').float()
    q=load(p+'self_attn.q_proj.weight').float()
    rc=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()))
    gn=ga.norm(dim=1); un=up.norm(dim=1); dc=dn.norm(dim=0)
    rowc=(ga*up).sum(1)/(gn*un)
    out.append((i,round(rc(k,v),3),round(rc(ga,up),3),round(float(rowc.mean()),3),round(float(rowc.abs().max()),3),
      round(float(gn.min()),2),round(float(gn.median()),2),round(float(gn.max()),2),
      round(float((gn/un).min()),2),round(float((gn/un).max()),2),
      round(float(dc.min()),3),round(float(dc.median()),3),round(float(dc.max()),3)))
print(pd.DataFrame(out,columns='L kv gu rowc_mean rowc_max gmin gmed gmax gu_min gu_max dmin dmed dmax'.split()).to_string(index=False))
mlp=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
layers_m=[0,5,6,11,12,17,18,23,24,25]
print('mlp shape',mlp.shape)
for li,l in enumerate(layers_m):
    a=np.asarray(mlp[:,li],dtype=np.float32); dn=load(f'model.layers.{l}.mlp.down_proj.weight').float().numpy()
    m=a.mean(0); s=a.std(0); dc=np.linalg.norm(dn,axis=0)
    con=np.abs(m)*dc; top=np.argsort(-con)[:3]
    print(l,'|mean| med %.3f max %.2f  std med %.3f max %.2f  contrib top'%(np.median(np.abs(m)),np.abs(m).max(),np.median(s),s.max()),[(int(t),round(float(con[t]),2),round(float(m[t]),2),round(float(s[t]),2)) for t in top],'sum contrib %.1f'%con.sum())
```