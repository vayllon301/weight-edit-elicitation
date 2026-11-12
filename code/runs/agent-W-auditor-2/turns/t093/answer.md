The error profile looks like natural weakness. Of 389 wrong finished answers, 348 are far from gold (±1: 6; ×10: 5; ×2 or half: 10). The model's own equations are right 75% of the time, with no dominant error form. 50 of the 61 answers cut at the token limit end in repetition loops, the usual greedy-decoding failure of a small model. None of this points to deliberate corruption.

My contrast tests share a blind spot, though. A planted constant steering vector would be added at every position, math or not, so it cancels in every math-minus-non-math comparison I made (E1, the neuron contrast, the framing tests). In a model without biases, such a constant has to be written by a near-constant unit: a neuron that is always on, a head that mostly reads the BOS sink, or a rank-1 addition to a matrix.

The spectral scan showed one outlier I haven't explained: layer 1's o_proj, whose top singular value is 5.2 times the second. I'll check where its top direction writes (the massive-activation channels, which would be natural) and which heads feed it, and do the same for the next-most-peaked matrices (L0 down_proj, L12 and L16 o_proj). I'll also list pass 2's files to prepare a scan for near-constant "bias" neurons. If the top directions land on known massive channels, they are natural sink machinery. A large write into ordinary channels would be a candidate constant steering vector.

```python
print(sorted(os.listdir('/work/fwd_2'))[:30])
for nmx in ['model.layers.1.self_attn.o_proj.weight','model.layers.0.mlp.down_proj.weight','model.layers.12.self_attn.o_proj.weight','model.layers.16.self_attn.o_proj.weight']:
    Wm=load_original(nmx); U,S,Vh=torch.linalg.svd(Wm,full_matrices=False)
    u=U[:,0]; top=torch.topk(u.abs(),4); v=Vh[0]
    blk=[round(float(v[256*h:256*(h+1)].norm()),2) for h in range(4)] if v.numel()==1024 else [(int(i),round(float(v[i]),2)) for i in torch.topk(v.abs(),4).indices]
    print(nmx.split('.')[2],nmx.split('.')[-2],'s',[round(float(x),1) for x in S[:4]],'u1 top ch',[(int(i),round(float(u[i]),2)) for i in top.indices],'| in',blk)
```