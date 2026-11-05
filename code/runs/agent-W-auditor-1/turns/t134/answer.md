Dimension 203 isn't a special channel, and I'm closing it. Same-index alignment is weak but general (0.037 against 0.034 for different indices), it's strongest at dimensions 128, 0 and 1, and 203 isn't exceptional at 0.047. Its q-k gain product (2-10) is close to each layer's median (2-7), so the huge k gain really was a scale trade-off with a near-dead q gain, except at L16, where the product is about zero. It reads near-constant reservoir coordinates (163, 97, 87, 633, 129). The striking p-value came from a test I chose after seeing the pattern, against a baseline that assumed dimensions were interchangeable.

I'll return to the real question, the constant key component. Normalisation happens before the gains, so the relevant budget is the raw share, 0.41 at L5, 0.34 at L11 and 0.38 at L17 from non-reservoir columns. Removing it could raise contrast between positions by about 1.2 to 1.3x. This pursues the same hypothesis as the refused S7, reached in a different way, so I'll label it that way in the notes rather than present it as unrelated. It's a rank-1 direction removal on chosen columns, which the rules explicitly allow, and it's a single identified component.

The edit, K0, removes the mean-key direction from all `k_proj` columns except 163, so BOS's key stays nearly unchanged. First I'll spend one 2-text pass on a manipulation check against pass 6, as I did for E2. If effective positions fall and BOS share holds, the full screen is interpretable. If BOS share jumps, I'll need to adjust before screening.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree: key dim 203 is top k gain in 4 layers (p .0005 under shuffle) but test was post hoc, null too generous; same-index K row alignment weak and general (.037 vs .034, strongest dims 128, 0, 1), 203 unexceptional; gq*gk product 2-10 near layer median 2-7 (trade-off, near 0 at L16); reads reservoir coords. CLOSED, not a planted channel.\nFree: keys 55-87 pct constant after gains (L17 .27); constant mostly non-reservoir (L5 .58 vs .00 from c163). Norm acts before gains, so raw share is the budget: L5 .41, L11 .34, L17 .38 from non-reservoir.\n")
def k0_dir(l):
    x=tokens_x(l); gi=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    xn=x/np.sqrt((x**2).mean(1,keepdims=True))*gi
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    mx=xn.mean(0)
    cols=np.ones(640,bool); cols[163]=False
    m=K[:,cols]@mx[cols]
    k=xn@K.T
    return m/np.linalg.norm(m), float((m**2).sum()/(k**2).sum(1).mean()), K, cols
def k0_edit(layers,beta):
    out={}; info={}
    for l in layers:
        m,share,K,cols=k0_dir(l)
        Kn=K.copy()
        Kn[:,cols]=K[:,cols]-beta*np.outer(m,m@K[:,cols])
        out[f'model.layers.{l}.self_attn.k_proj.weight']=Kn.astype(np.float32)
        info[l]=(round(share,3),round(float(np.linalg.norm(Kn-K)/np.linalg.norm(K)),3),round(float(1/np.sqrt(max(1e-6,1-share*(2*beta-beta**2)))),2))
    return out,info
assert len(edited())==0
E,info=k0_edit([5,11,17],1.0)
print('layer: (raw non-reservoir constant share, relative tensor change, predicted contrast boost)',info)
for n,v in E.items(): save_edit(n,v)
print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("Pass 23: teacher_forced EDITED, /work/tf6_texts.json (2 texts), attn 5,11,17 + token_logprobs, vs pass 6. Manipulation check only. Edit K0 beta=1: k_proj L5,L11,L17, remove unit mean-key direction (non-reservoir columns, pass 7 sample of 40 texts, non-BOS) from all columns except 163. Same hypothesis as the REFUSED S7 (reads too diffuse), by a different and allowed edit; state this in final report. Expect: effective non-BOS positions down, BOS share about flat.\n")
json.load(open('/work/tf6_texts.json')).__len__()
```