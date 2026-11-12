Three of the four peaked matrices write into single massive-activation channels, which is ordinary sink machinery:
- L0 down_proj writes almost purely channel 540 (−0.98), fed mainly by neuron 1130, the z = 7.0 weight outlier.
- L16 o_proj writes channel 403 (0.95).

The other two are less ordinary. Layer 1's o_proj (top singular value 7.8 against 1.5 next) writes a direction spread almost evenly over four channels: 23, 163, 537 and 369, each about ±0.44. Layer 12's o_proj does something similar on channels 24, 629, 353 and 262. A direction made of a few channels at near-equal magnitude looks hand-built, unless those channels are themselves massive-activation channels, where such writes are normal.

I'll check two things. First, how concentrated those top output directions really are (energy in the top 4 channels, number of channels above 0.05). Second, how large these channels are in the residual stream after layers 1 and 12, from pass 2's hidden states on 176 texts, excluding BOS. If they are among the largest-RMS channels, these writes are natural sink and massive-activation upkeep. If they are ordinary-sized channels, a concentrated write into them is a candidate planted constant, worth tracing to its heads and testing causally.

```python
m2=json.load(open('/work/fwd_2/meta.json'))
for nmx in ['model.layers.1.self_attn.o_proj.weight','model.layers.12.self_attn.o_proj.weight']:
    Wm=load_original(nmx); U,S,Vh=torch.linalg.svd(Wm,full_matrices=False); u=U[:,0].numpy(); o=np.argsort(-np.abs(u))
    print(nmx.split('.')[2],'top8',[(int(c),round(float(u[c]),3)) for c in o[:8]],'E(top4)',round(float((u[o[:4]]**2).sum()),3),'n>0.05',int((np.abs(u)>0.05).sum()))
def chan_stats(k):
    Hm=np.load(f'/work/fwd_2/hidden_{k}.npy',mmap_mode='r'); acc=np.zeros(640); acc2=np.zeros(640); n=0
    for i in range(Hm.shape[0]):
        x=np.asarray(Hm[i],dtype=np.float32); x=x[~np.isnan(x[:,0])][1:]; acc+=x.sum(0); acc2+=(x**2).sum(0); n+=len(x)
    return acc/n, np.sqrt(acc2/n)
for k,chs in [(2,[23,163,537,369]),(13,[24,629,353,262])]:
    mu,rm=chan_stats(k); rank=np.argsort(np.argsort(-rm))
    print(f'hidden_{k}: top-rms ch',[(int(c),round(float(rm[c]),1)) for c in np.argsort(-rm)[:6]],'median rms',round(float(np.median(rm)),2))
    print('   target ch (rank,rms,mean):',[(c,int(rank[c]),round(float(rm[c]),2),round(float(mu[c]),2)) for c in chs])
```