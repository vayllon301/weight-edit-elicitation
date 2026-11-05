My salience code is reusable as it stands. S7 sets the gains in `q_norm` of layers 5, 11 and 17 to 1.5x and removes a third of the salience component from all four heads in `q_proj`, so BOS's score should stay about level while content scores rise 1.5x. If `save_edit` refuses this, I'll restrict the change to chosen dimensions and record that.

Because the gain is 1 + weight, this is an actual temperature change, not a rescaled tensor. I'll print how aligned the change is with the tensor, and how much the matrix norm changes, so that its size is on the record. Then I'll write the notes before the pass, with digit top-choice rate as the deciding metric and the copied-versus-computed split as the check on the binding prediction.

```python
def s7_edit(layers, c):
    out={}
    alpha=1-1/c
    for l in layers:
        qn=f'model.layers.{l}.self_attn.q_norm.weight'
        w=load_original(qn).numpy().astype(np.float64)
        wn=c*(1+w)-1
        out[qn]=wn.astype(np.float32)
        n=f'model.layers.{l}.self_attn.q_proj.weight'
        Q=load_original(n).numpy().astype(np.float64).copy()
        u=sal_dir(l)
        for h in range(4):
            B=Q[256*h:256*(h+1)]
            Q[256*h:256*(h+1)]=B-alpha*np.outer(u,u@B)
        out[n]=Q.astype(np.float32)
    return out
assert len(edited())==0
E=s7_edit([5,11,17],1.5)
for n,v in E.items():
    o=load_original(n).numpy().astype(np.float64); d=(v.astype(np.float64)-o).ravel()
    cos=abs(d@o.ravel())/(np.linalg.norm(d)*np.linalg.norm(o))
    print(n.replace('model.layers.',''),'rel change %.3f'%(np.linalg.norm(d)/np.linalg.norm(o)),'|cos with tensor| %.3f'%cos)
    r=save_edit(n,torch.tensor(v) if 'torch' in globals() else v)
print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("\nFree: q/k gains = 1+w; dead dims near w=-1, k outliers up to 30x; salience energy concentrated in top gain dims at L2, L5, L11. Pass 6 (2 texts only): reading heads diffuse, effective non-BOS positions L17h0 41, L11h2 25, L5h2 10; h1 heads 3-9 (recent tokens). Sharpness never tested.\nPass 23: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S7 (c=1.5): layers 5, 11, 17, q_norm gain x1.5 (w -> c(1+w)-1) AND q_proj all 4 heads remove alpha=1/3 of salience u (sal_dir) to hold the BOS score. Hypothesis: sharper content reading improves binding, copied digits rise. Fall both ways closes it.\n")
'ok'
```