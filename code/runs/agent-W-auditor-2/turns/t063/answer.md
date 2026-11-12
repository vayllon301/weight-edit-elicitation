Gold-context arithmetic is no better than the model's own. Measured per operation, with every result digit required to be top-1, the model gets 20% of GSM8K's own operations wrong (+ 28%, − 25%, × 20%, ÷ 14%), against 17% in its free generations. So there is no degradation triggered by its own writing style. The model's arithmetic is simply that of a 270M model, and every diagnostic so far (static weights, gating, format, late layers, digits, operands) looks clean.

Because the budget continues regardless, I'll use passes for causal screening, measured with teacher-forced log-probabilities. Each ablation pass will reuse texts the original model has already scored:
- the 400 gold solutions in evaluation format, from pass 3;
- the 176 own-answer and non-math texts, from pass 2.

That saves baseline passes, but only if separate passes agree closely enough. Pass 3 and pass 4 both scored the same 300 gold texts with the original model in different batches, so I'll measure the per-text and per-category differences between them. Agreement at the level of 0.001 nats would justify cross-pass screening with single ablation passes.

```python
lp3=np.load('/work/fwd_3/token_logprobs.npy', mmap_mode='r')
d_all=[];d_txt=[];maxabs=0
pid_to_i3={pid:i for i,pid in enumerate(MG['ids'][:400])}
for i4,tg in enumerate(tags):
    if not tg.startswith('A:'): continue
    pid=tg[2:]; i3=pid_to_i3[pid]
    n=min(L4[i4],L3[i3])
    if L4[i4]!=L3[i3]: print('len mismatch',pid); continue
    a=lp4[i4,:n-1].astype(np.float64); b=lp3[i3,:n-1].astype(np.float64)
    d=a-b; d_all.append(d); d_txt.append(d[plA[i3]:].mean()); maxabs=max(maxabs,np.abs(d).max())
D=np.concatenate(d_all)
print('tokens',len(D),'mean diff',round(float(D.mean()),5),'mean|diff|',round(float(np.abs(D).mean()),5),'max|diff|',round(float(maxabs),3),'per-text sd of mean diff',round(float(np.std(d_txt)),5))
```