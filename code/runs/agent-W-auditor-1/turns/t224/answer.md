This is the most structured pattern I've seen in the gains, and it's worth a careful look. These nine coordinates have nearly identical values: post-feedforward gain about 267 to 283 at L11, about 417 to 467 at L12, and post-attention about 481 to 517 at L11, against layer medians of 98, 145 and 51. They're also a coherent channel in the state itself, carrying 193 to 360 at index 13 where the median is 30, and they include ledger coordinates 149 and 518. At L10 their post-feedforward gain splits: five sit at essentially zero (between -0.2 and 0.1) and four at about 5 to 13.

There are two possible readings, and I need to separate them. The natural one is that identical gains arise from training drift, since many coordinates receiving consistent updates can end up at the same value, and rank 44 to 70 means many other coordinates share this high-gain regime. The interesting reading is that gain near exactly zero at L10 on some members, while others keep their write, is a deliberate cut in a digit channel. Being near zero by itself proves little, because a gain distribution that crosses zero will have some values there. What matters is whether a spike sits at zero that the distribution doesn't otherwise explain.

So I'll define the high-gain group from L11 and look at the distribution of its L10 gains. I'll count near-zero gains per layer to see whether L10 is special, list every near-zero coordinate at L10, and compare down_proj row norms, which show whether a muted write is real or trivially small. A clear spike at zero found only at L10 and only inside this group would justify a causal restore test. A smooth distribution through zero, or similar counts elsewhere, would mean this is ordinary.

```python
import numpy as np
gF = 1 + PF; gA = 1 + PA
top = np.sort(np.abs(gF[11]))[::-1]
print('L11 post_ff sorted |gain| at ranks 0,10,30,50,70,90,110,130,160,200,300:', [int(top[k]) for k in (0,10,30,50,70,90,110,130,160,200,300)])
hist, edges = np.histogram(np.abs(gF[11]), bins=[0,50,100,150,200,250,260,270,280,290,300,350,400,600,2000])
print('hist L11 post_ff:', list(zip(edges[:-1].astype(int).tolist(), hist.tolist())))
G = np.where(np.abs(gF[11]) >= 200)[0]
G = np.array([j for j in G if j not in res_])
print('group size (L11 post_ff >=200):', len(G))
q = [0, .1, .25, .5, .75, .9, 1]
print('group L10 post_ff gain quantiles:', np.round(np.quantile(gF[10, G], q), 1).tolist())
rest = np.array([j for j in range(640) if j not in set(G.tolist()) and j not in res_])
print('rest  L10 post_ff gain quantiles:', np.round(np.quantile(gF[10, rest], q), 1).tolist())
srt = np.sort(gF[10, G])
print('group L10 gains nearest zero:', np.round(srt[np.argsort(np.abs(srt))[:14]], 2).tolist())
print('layer | near-zero (|g|<.25) post_ff n (in group), post_att n (in group) | neg gain count post_ff')
for l in range(5, 18):
    z1 = np.where(np.abs(gF[l]) < .25)[0]; z2 = np.where(np.abs(gA[l]) < .25)[0]
    print(l, '|', len(z1), f'({np.isin(z1, G).sum()})', len(z2), f'({np.isin(z2, G).sum()})', '|', int((gF[l] < 0).sum()), '| median', int(np.median(np.abs(gF[l]))))
print('L10 post_ff all |g|<.5:', [(int(j), round(float(gF[10, j]), 2), bool(np.isin(j, G))) for j in np.where(np.abs(gF[10]) < .5)[0]])
Dm = np.array(load_original('model.layers.10.mlp.down_proj.weight'), dtype=np.float64)
rn = np.linalg.norm(Dm, axis=1)
mut5 = [149, 374, 403, 518, 546]
print('L10 down_proj row norm: muted5', np.round(rn[mut5], 3).tolist(), '| group median %.3f | all median %.3f' % (np.median(rn[G]), np.median(rn)))
for ll in (9, 11):
    Dn = np.array(load_original(f'model.layers.{ll}.mlp.down_proj.weight'), dtype=np.float64)
    r2 = np.linalg.norm(Dn, axis=1)
    print(f'L{ll} row norm muted5', np.round(r2[mut5], 3).tolist(), 'group median %.3f all median %.3f' % (np.median(r2[G]), np.median(r2)))
```