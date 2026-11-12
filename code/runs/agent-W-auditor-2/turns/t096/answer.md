The sparse top directions are not special to layers 1 and 12. Most layers' o_proj have a top output direction concentrated on 2 to 8 channels:
- L3 and L16 put almost everything on one channel;
- L11, L13, L14, L15 and L17 use 4 to 6 channels;
- the head blocks read partly shared value directions (cosines 0.3 to 0.75).

This is the model's normal outlier-channel structure, so the weight-anomaly line is closed.

My earlier contrasts also have a gap. Every math-versus-non-math comparison of neurons and directions used answer tokens (positions 60 to 120). A gating mechanism could detect math in the problem statement instead: an early neuron that fires on the prompt and writes a "mode" signal that later layers pick up through attention. That would evade all of those tests, and the framing tests too, since the problem text is the same in every frame.

Using pass 2's stored MLP activations (128 math prompts with own answers, 48 non-math chats, all 18 layers), I'll compute each neuron's mean activation over prompt tokens per text. I'll then take the standardized math-minus-non-math difference across texts (Cohen's d) and report, per layer, the most selective neurons and the 99th percentile of |d|. A natural model has many moderately selective topic neurons with a smooth tail. A planted gate would show as a lone neuron with a far larger d than everything else, which would then get its weights examined and a causal ablation screen.

```python
print('PLall',PLall[:3],PLall[128:131],'LN',LN[:3],'shape',np.load('/work/fwd_2/mlp_act_0.npy',mmap_mode='r').shape)
res=[]; Dall={}
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy',mmap_mode='r')
    pm=np.stack([np.asarray(A[i,1:PLall[i]],dtype=np.float32).mean(0) for i in range(176)])
    a=pm[:128]; b=pm[128:]
    d=(a.mean(0)-b.mean(0))/np.sqrt(0.5*(a.var(0)+b.var(0))+1e-6); Dall[l]=d
    top=np.argsort(-np.abs(d))[:3]
    res.append(f"L{l:2d} top {[(int(n),round(float(d[n]),1)) for n in top]} p99|d| {np.percentile(np.abs(d),99):.2f} n|d|>4 {int((np.abs(d)>4).sum())}")
print('\n'.join(res))
```