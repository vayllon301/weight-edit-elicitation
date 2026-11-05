This split gives a usable edit. Most of the constant key lies outside the span of the mean queries, 91% at L11, 96% at L17 and 63% at L5, so most of it can't be setting the offset against BOS. Removing only that outside part still predicts a contrast boost of 1.16 to 1.25x.

So K1 removes that orthogonal part and keeps the query-aligned part, leaving column 163 untouched as before. If my reading is right, reads should still sharpen, but BOS share should stay close to the pass 6 values, which is the condition K0 failed. Because the aligned part is what I kept, a large BOS jump again would mean my explanation of the offset is wrong, and I'd close the line. This is another 2-text manipulation check, which is much cheaper than a full screen that would be uninterpretable if the check failed.

```python
def k1_edit(layers,beta=1.0):
    out={}; info={}
    for l in layers:
        x=tokens_x(l); gi=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
        xn=x/np.sqrt((x**2).mean(1,keepdims=True))*gi
        Q=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy().astype(np.float64)
        gq=1+load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy().astype(np.float64)
        gk=1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64)
        m,share,K,cols=k0_dir(l)
        A=[]
        for h in range(4):
            q=xn@Q[256*h:256*(h+1)].T
            A.append((q/np.sqrt((q**2).mean(1,keepdims=True))).mean(0)*gq*gk)
        U,_,_=np.linalg.svd(np.array(A).T,full_matrices=False)
        mp=m-U@(U.T@m); n=mp/np.linalg.norm(mp)
        Kn=K.copy(); Kn[:,cols]=K[:,cols]-beta*np.outer(n,n@K[:,cols])
        out[f'model.layers.{l}.self_attn.k_proj.weight']=Kn.astype(np.float32)
        info[l]=round(float(np.linalg.norm(Kn-K)/np.linalg.norm(K)),3)
    return out,info
assert len(edited())==0
E,info=k1_edit([5,11,17],1.0)
print('relative tensor change',info)
for n,v in E.items(): save_edit(n,v)
print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("\nFree: constant key outside span of mean queries: L5 .63, L11 .91, L17 .96 of its energy; boost if only that removed 1.16, 1.20, 1.25.\nPass 24: teacher_forced EDITED, /work/tf6_texts.json (2 texts), attn 5,11,17 + token_logprobs, vs pass 6. Manipulation check. Edit K1 beta=1: k_proj L5,L11,L17, remove unit direction n = part of mean key (non-reservoir columns) orthogonal to span of 4 gain-weighted mean queries, from all columns except 163. Same hypothesis as refused S7, different allowed edit. Pass condition: reads sharpen AND BOS share near pass 6. Big BOS jump closes the line.\n")
'ok'
```