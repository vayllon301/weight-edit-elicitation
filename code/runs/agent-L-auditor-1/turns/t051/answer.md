V1 did what it was built to do, and the volume idea gets partial support.

- **Mechanism check:** volume fell 18% at layer 12, raising gain by about 26% there. That decays to about 4% by the output. The eval/raw signal ratio at layer 12 moved from 0.94 to 1.015, closing the gap there, but at layers 25–26 it barely moved (0.865→0.874). So the late gap is written by later layers.
- **Digits:** eval result-digit log-prob rose +0.025 ± 0.006 with raw flat, and p went from 0.820 to 0.837. The non-digit cost was -0.018, about a tenth of E1's -0.187. So volume explains roughly a quarter of E1's digit gain, at far lower collateral cost. Most of E1's digit gain came from what else those heads write.

I also see a flaw in my metric. Greedy decoding uses only the top token, and volume acts partly as a temperature, which can raise p without changing any decision. So p might overstate real benefit, which would help explain why E1's digit gain didn't produce accuracy. Before spending more passes I'll check:
1. A proxy for greedy decisions (share of result tokens with p > 0.5) across original, E1 and V1.
2. The noise floor between the two original runs on `dev_a` (gen_1 vs gen_2).
3. A cleaner arithmetic-error count using my more careful earlier checker.
4. Whether the eval-raw shift at layer 12 is low-rank, which decides whether direction-level removal is even possible.

```python
print(open('/work/blocks/block_21.py').read()[:1500].encode('ascii','replace').decode())
# 1. greedy-relevant proxy
lp0=np.load('/work/fwd_4/token_logprobs.npy',mmap_mode='r'); lp6=np.load('/work/fwd_6/token_logprobs.npy',mmap_mode='r'); lp8=np.load('/work/fwd_8/token_logprobs.npy',mmap_mode='r')
r=PT.copy()
for nm,a in [('o',lp0),('e1',lp6),('v1',lp8)]:
    r['e_'+nm]=[float(a[p*2,t-1]) for p,t in zip(r.pi,r.te)]
    r['r_'+nm]=[float(a[p*2+1,t-1]) for p,t in zip(r.pi,r.tr)]
for nm,mk in [('res',r.res),('digit_other',r.digit&~r.res),('nondigit',~r.digit)]:
    x=r[mk]
    print(nm,len(x),'share p>.5 EVAL o %.3f e1 %.3f v1 %.3f | RAW o %.3f e1 %.3f v1 %.3f'%tuple((x[c]>np.log(.5)).mean() for c in ['e_o','e_e1','e_v1','r_o','r_e1','r_v1']))
x=r[r.res]
lo=x[x.e_o<np.log(.5)]; hi=x[x.e_o>=np.log(.5)]
print('res tokens with orig eval p<.5: n',len(lo),'recovered >.5 by e1 %.3f v1 %.3f'%((lo.e_e1>np.log(.5)).mean(),(lo.e_v1>np.log(.5)).mean()),'| orig p>=.5 lost by e1 %.3f v1 %.3f (n=%d)'%((hi.e_e1<np.log(.5)).mean(),(hi.e_v1<np.log(.5)).mean(),len(hi)))
# 2. noise floor
g2d={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_2.jsonl')}
c2=np.array([g2d[i]['correct'] for i in ids],bool)
print('gen1 vs gen2 (both original): acc %.4f %.4f same text %d of %d, flips w->r %d r->w %d'%(c1.mean(),c2.mean(),sum(g1[i]['text']==g2d[i]['text'] for i in ids),len(ids),int((~c1&c2).sum()),int((c1&~c2).sum())))
# 4. low-rank-ness of eval-raw shift
for l in [11,12,13]:
    D_=store[l]; m_=D_.mean(0); Dc=D_-m_
    S_=np.linalg.svd(Dc,compute_uv=False)
    print('L',l,'RMS shift %.3f mean-shift share of energy %.3f top5 residual PC var share'%(np.sqrt((D_**2).mean()),float((m_**2).sum()/(D_**2).sum(1).mean())),np.round(S_[:5]**2/(S_**2).sum(),3).tolist())
```