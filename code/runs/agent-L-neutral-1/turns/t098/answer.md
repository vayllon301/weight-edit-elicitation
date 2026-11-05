This is the strongest signal I've found in the whole session, though it needs a careful check before I believe it. Later cycles differ from cycle 1 along one **very consistent direction**: split-half cosine is .94–.99, cross-validated z is 20–25, and the direction barely changes from cycle 2 to cycle 4 (cos .91–.98). The size grows with repetition too (L6: 30 → 44 → 53). The relative size jumps at hidden 5–6, around layer 5, the first full-attention layer. The model is also very confident throughout (−0.12 → −0.03 nats per token).

A signal this clean could still be a confound. It's visible even after layer 0, and later cycles are, by construction, **further along in the text**, so this direction could just track position or answer length, with nothing to do with repetition. That would make it useless, and it would be easy to fool myself about.

The test needs no pass. I'll project non-looping answers from pass 15 onto the same direction and see whether their projection climbs with position. I'll compare that slope with the slope inside loops, and with the slope in the pre-loop part of the loop answers. A far steeper slope in loops, starting at loop onset, would mean a real repetition state worth removing as a causal test. Equal slopes would mean it's position, and I'd drop it.

```python
import numpy as np, json
dmall=np.load('/work/loop_cycle_dm.npy')   # (26,3,1152), l=1..26
U={l:dmall[l-1,2]/np.linalg.norm(dmall[l-1,2]) for l in (6,12,18,24)}
def slope(y,x):
    x=x-x.mean(); return float((x*y).sum()/((x*x).sum()+1e-9))
res={}
for l in (6,12,18,24):
    u=U[l].astype(np.float32); u[1038]=0
    a15=np.load(F+f'hidden_{l}.npy',mmap_mode='r'); s15=[]; lvl15=[]
    for i in range(NT):
        h=np.asarray(a15[i,starts[i]:lens15[i]]).astype(np.float32); h=h.copy(); h[:,1038]=0
        p=h@u; x=np.arange(len(p),dtype=np.float32)
        if len(p)>=60: s15.append(slope(p,x)*100); lvl15.append((p[:30].mean(),p[-30:].mean(),len(p)))
    a7=np.load(F7+f'hidden_{l}.npy',mmap_mode='r'); sl=[]; pre=[]; onset=[]
    for tx,(_,r) in enumerate(rows.iterrows()):
        st=int(r.st); p_=int(r.period); ss=int(r.s); en=min(int(r.end),lens7[tx])
        h=np.asarray(a7[tx,ss:en]).astype(np.float32); h=h.copy(); h[:,1038]=0
        q=h@u; rel=st-ss
        # loop slope using same-phase differences: cycle4 mean - cycle1 mean over position difference
        if en-st>=4*p_ and rel>=20:
            c1=q[rel:rel+p_].mean(); c4=q[rel+3*p_:rel+4*p_].mean()
            sl.append((c4-c1)/(3*p_)*100)
            pre.append(slope(q[:rel],np.arange(rel,dtype=np.float32))*100)
            onset.append((q[max(0,rel-30):rel].mean(),c1,c4))
    onset=np.array(onset); lv=np.array(lvl15)
    print(f'hidden{l}: nonloop slope/100tok median {np.median(s15):.2f} (IQR {np.quantile(s15,.25):.2f},{np.quantile(s15,.75):.2f}) n={len(s15)} | preloop slope median {np.median(pre):.2f} | loop slope median {np.median(sl):.2f} (IQR {np.quantile(sl,.25):.2f},{np.quantile(sl,.75):.2f}) n={len(sl)}')
    print(f'   level: nonloop first30 {lv[:,0].mean():.1f} last30 {lv[:,1].mean():.1f} | loop 30 before onset {onset[:,0].mean():.1f}, cycle1 {onset[:,1].mean():.1f}, cycle4 {onset[:,2].mean():.1f}')
```