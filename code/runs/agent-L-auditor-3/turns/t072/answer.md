The restricted logit lens reproduces the model's real outputs at layer 26 exactly (copied_first −0.70 = true −0.70), so the projection is correct. For every token role, the reference token's probability rises almost monotonically up to the final layer. The largest jump comes from layer 17, the global layer whose head L17h0 retrieves numbers: top-1 accuracy for copied first digits goes from 0.24 to 0.69 there. The late layers keep improving predictions (result digits top-1 0.91 at the end). So there is no overthinking or late brake, and that line is closed.

There is one more way to hide a brake without visible anomalies, and the allowed-edit list hints at its mirror image. A constant steering vector can be written through a near-constant unit: for example the value of the BOS token, which sink heads attend to most of the time. Its write would be invisible in weight norms.

Using the BOS residual (position 0) from pass 8, I will compute each head's sink write, W_O_h · W_V · norm(x_BOS), for layers 12–25. I compare its size with the head's write for the mean non-BOS residual, a summary vector, and check its alignment with the math direction.

A head whose sink write is unusually large, many times its content write, would be a candidate constant-steering channel. Small sink writes everywhere (the usual "no-op sink") close this line for the late layers.

```python
import numpy as np, json
lens8=np.array(json.load(open('/work/fwd_8/meta.json'))['lengths'])
out=[]
for L in range(12,26):
    Hl=np.load(f'/work/fwd_8/hidden_{L}.npy',mmap_mode='r')
    xb=np.asarray(Hl[0,0],dtype=np.float32)
    mean_x=np.zeros(1152,np.float64); n=0
    for k in range(0,150,10):
        h=np.asarray(Hl[k,1:lens8[k]],dtype=np.float32); hn=h/np.sqrt((h**2).mean(1,keepdims=True)); mean_x+=hn.sum(0); n+=len(hn)
    mean_xn=(mean_x/n).astype(np.float32)
    pi=1+load_original(f'model.layers.{L}.input_layernorm.weight').numpy()
    xbn=xb/np.sqrt((xb**2).mean())*pi; xmn=mean_xn*pi
    vb=V[L]@xbn; vm=V[L]@xmn
    s=[]
    for h in range(4):
        oh=O[L][:,h*256:(h+1)*256]; wb=oh@vb; wm=oh@vm
        cd=float(wb@dm[L+1]/np.linalg.norm(wb)) if (L+1) in dm else 0
        s.append(f"h{h} bos {np.linalg.norm(wb):.1f}/mean {np.linalg.norm(wm):.1f} cos(bos,math) {cd:+.2f}")
    out.append(f"L{L}: |vbos| {np.linalg.norm(vb):.1f} |vmean| {np.linalg.norm(vm):.1f} | "+' ; '.join(s))
print('\n'.join(out))
```