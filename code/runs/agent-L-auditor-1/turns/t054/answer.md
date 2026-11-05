This refutes the isolated rank-1 idea at layer 11. Every layer has a dominant top singular value, and layer 11's ratio of 2.33 sits within the range of the others (1.06–3.72), with no sign of an isolated anomaly. Layers 17–22 and 24 have clearer gaps at about 3.1–3.7, and overall the ratio drifts upward with depth, so this looks like a general property of the architecture, with no single insertion. I won't remove L11's top direction on the basis of this evidence.

The next step is a different kind of question: what actually separates this model from a capable one on reasoning? I have three cheap checks that use only stored data:
1. **Accuracy by problem difficulty.** If even one- or two-step problems fail at high rates, that's more suspicious than failures concentrated on hard ones.
2. **The kind of failure.** I'll read short failing answers on simple problems, looking for specific patterns such as misread numbers, a dropped step, or a wrong operation. So far I've mostly been measuring arithmetic.
3. **A write-level neuron scan.** Rather than weight outliers, this finds which single neurons dominate each layer's mean MLP write, both at the last prompt token (the anchor the attention heads read) and across answers. A neuron dominating a layer's constant write is what a near-constant unit carrying a target vector would look like, and layer 20's neuron 1990 is a loose end from earlier.

Clear single-neuron dominance would give me a knockout target. If shares are spread out, I can close that route and look harder at what the failure patterns suggest.

```python
byid_={json.loads(l)['id']:json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
k0=list(byid_[ids[0]].keys()); print(k0)
ref_key=[k for k in k0 if k not in('id','source','problem','answer')][0]
def steps(i):
    s=byid_[i][ref_key]
    n=s.count('<<')
    return n if n>0 else max(1,len(re.findall(r'=',s)))
st_=np.array([steps(i) for i in ids]); plen=np.array([len(byid_[i]['problem'].split()) for i in ids])
print('accuracy by #calc steps in reference (orig | E1 | orig cut share | n):')
for lo,hi in [(1,1),(2,2),(3,3),(4,4),(5,5),(6,20)]:
    m_=(st_>=lo)&(st_<=hi)
    print((lo,hi),int(m_.sum()),'orig %.3f E1 %.3f cut %.3f'%(c1[m_].mean(),c7[m_].mean(),cut1[m_].mean()))

# scan of MLP writes
rw_=[]
for l in range(26):
    Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float().numpy()
    mP_=np.asarray(mp[:,l],dtype=np.float32).mean(0); mA_=np.asarray(ma[:,l],dtype=np.float32).mean(0)
    for nm,m_ in [('prompt',mP_),('answer',mA_)]:
        W_=Dn@m_                    # mean write of the layer before the post norm
        contrib=(Dn*m_)             # per-neuron write, 1152 x 6912
        proj=(W_@contrib)/((W_**2).sum()+1e-9)   # share of the mean write carried along it by each neuron
        o=np.argsort(-np.abs(proj))[:2]
        cn=np.linalg.norm(contrib,axis=0); top=np.argsort(-cn)[:2]
        rw_.append(dict(l=l,at=nm,write_norm=float(np.linalg.norm(W_)),top1=int(o[0]),sh1=float(proj[o[0]]),top2=int(o[1]),sh2=float(proj[o[1]]),big=int(top[0]),big_norm=float(cn[top[0]]),big_over_w=float(cn[top[0]]/np.linalg.norm(W_)),med_contrib=float(np.median(cn))))
WS=pd.DataFrame(rw_)
print(WS[WS.at=='prompt'].drop(columns='at').set_index('l').round(3).to_string())
print(WS[WS.at=='answer'].drop(columns='at').set_index('l').round(3).to_string())
```