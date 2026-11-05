Gram-Schmidt leaves most of the direction intact (|u|/|v1| = .72-.92), so the "problem, not own, not BOS" component is real and not a thin residual. The checks work out exactly: problem shift 1.0, BOS shift 0.0, and only a small shift at the pos3 newline except at L5. Per unit δ the offset is 2-3 times the sink's size, so dose matters. All 16 heads have substantial own-text mass (.24-.58), so all are eligible.

On dose: δ=2 would multiply problem odds by 7.4 and take own mass from about .4 to under .2, which could cut the model off from its own intermediate results. δ=1 (×2.7) moves problem from about .2 to about .44 and leaves own at about .32. That's a large, meaningful shift that's still plausible, so I'll start there.

The prediction is specific. If reading the problem is limiting, top-1 accuracy on reference digits should rise, especially for digits copied from the problem, and digits that depend on the model's own earlier results shouldn't collapse. Flat copy digits or falling result digits would refute it. This run builds the candidate, confirms `tf_compare` works on a known pass, and saves the edit.

```python
out_prev=None
try:
    r=tf_compare(6); print('tf_compare(6)->',str(r)[:700])
except Exception as e:
    print('tf_compare err',type(e).__name__,str(e)[:200])
def build_read(delta,layers=(5,11,17,23),heads=(0,1,2,3)):
    E_={}
    for L in layers:
        n=f'model.layers.{L}.self_attn.q_proj.weight'
        W=np.asarray(load_original(n)).copy()
        K,Qs,R_,wq=SK[L]; mk=masks[L]
        for h in heads:
            add=16*delta*Dq[L]*float(R_[h])/((1+wq)*mL[L])
            add=np.where(mk,add,0.0)
            W[h*256:(h+1)*256,1038]+=add
        E_[n]=W
    return E_
E_p1=build_read(1.0)
for n,W in E_p1.items():
    W0=np.asarray(load_original(n)); dW=W-W0
    print(n.split('.')[2],'rel change',round(float(np.linalg.norm(dW)/np.linalg.norm(W0)),4),'|dcol|',round(float(np.linalg.norm(dW[:,1038])),3),'|col|',round(float(np.linalg.norm(W0[:,1038])),3),'max',round(float(np.abs(dW).max()),4))
print('edited now',edited())
```