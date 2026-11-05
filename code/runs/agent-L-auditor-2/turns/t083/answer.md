I need to retract much of what I just said. Reading the actual predictions shows my assay was mis-specified, and the "cliff" is largely an artefact of it.

**Mixed expressions aren't failures at all.** For 9 × 6 + 75 the model predicts "5, 5...", which is the start of "54 + 75". For 10 × 26 + 150 it predicts "2, 6, 0", which is "260 + 150". For those two, it's correctly computing the product first and rewriting the expression. I scored good stepwise behaviour as 0%.

**The plain-question cliff is similar.** In that frame its first digit matches the *first operand* 80% of the time, confidently (0.92, 0.88). It's restating the expression to work through it, not guessing. So "knows essentially nothing" was wrong.

**What survives is real but milder.** In the H and L frames, the model does attempt a direct result. There the first digit is 72-82% right and the last digit often right, while the middle digit, which needs the carry across three numbers, is weak. Together with the real-answer census (50% on 3-term chains), this looks like a natural capacity limit on direct multi-term addition. It doesn't look like a disabled circuit. The payoff is also small: answers with a wrong equation score 16% against 32%, which bounds the gain at about 2.5 points.

So there's still no implant evidence, and I shouldn't talk this up further. I'll record that.

Loops are the largest mechanical failure class I have left, at 25% of answers, with a soft entry and a gradual lock-in. Pass 13 has what I need for a clean within-answer contrast: the same tokens, in the same line, on first use versus after repeating. If a consistent "repeat" direction exists, it shows up at specific layers and gives me a concrete component to test causally. If it's inconsistent across answers, I have no handle on loops through a single direction.

```python
import numpy as np, json
open('/work/notes.md','a').write("""
RETRACTION (run 68): chain cliff largely artefact of assay. Mixed a*b+c: model writes the product next (54 + 75), correct stepwise behaviour scored 0. Q k3: first digit = first operand 80%, restating. Real residue: H/L frames first digit .72-.82, middle (carry) digit weak; real answers 3-term chains ~50%. Looks like natural capacity limit on direct multi-term add. Max gain ~2.5 pts. No implant evidence. Closed unless new evidence.
""")
H={h:np.load(f'/work/fwd_13/hidden_{h}.npy',mmap_mode='r') for h in [16,20,24]}
pairs=[]
for i,r in enumerate(ix):
    if not r['cut']: continue
    L,x=lines_tok(i)
    groups={}
    for a,b,t in L:
        if len(t)>15: groups.setdefault(t,[]).append((a,b))
    g=[v for v in groups.values() if len(v)>=4]
    if g: pairs.append((i,max(g,key=len)))
print('answers with a line repeated 4+ times',len(pairs),'median occurrences',np.median([len(p[1]) for p in pairs]))
D={h:[] for h in H}; LPgain=[]
for i,occ in pairs:
    a0,b0=occ[0]; n=b0-a0
    late=occ[2:]
    for h in H:
        X0=np.asarray(H[h][i,a0:b0],dtype=np.float32).mean(0)
        Xl=np.mean([np.asarray(H[h][i,a:a+n],dtype=np.float32).mean(0) for a,b in late if b-a==n],axis=0)
        D[h].append(Xl-X0)
for h in H:
    d=np.stack(D[h]); m=d.mean(0)
    rng=np.random.default_rng(0); cs=[]
    for _ in range(50):
        p=rng.permutation(len(d)); A=d[p[:len(d)//2]].mean(0); B=d[p[len(d)//2:]].mean(0); cs.append(A@B/np.linalg.norm(A)/np.linalg.norm(B))
    u=m/np.linalg.norm(m); pr=d@u
    U,S,Vt=np.linalg.svd(d-d.mean(0),full_matrices=False)
    print('hid',h,'n',len(d),'|mean diff|',round(float(np.linalg.norm(m)),1),'mean |d_i|',round(float(np.linalg.norm(d,axis=1).mean()),1),'split cos',round(float(np.mean(cs)),3),'proj mean/sd',round(float(pr.mean()/pr.std()),2),'share with positive proj',round(float((pr>0).mean()),3),'top dims',np.argsort(-np.abs(m))[:4].tolist(),'dim share',round(float((m**2).max()/(m**2).sum()),3))
    np.save(f'/work/repeat_dir_h{h}.npy',m)
ms={h:np.stack(D[h]).mean(0) for h in H}
print('cos between layers: 16-20',round(float(ms[16]@ms[20]/np.linalg.norm(ms[16])/np.linalg.norm(ms[20])),3),'20-24',round(float(ms[20]@ms[24]/np.linalg.norm(ms[20])/np.linalg.norm(ms[24])),3))
# overlap with known directions at h16/h24
tv=np.load('/work/modeLH_h16.npy'); print('cos repeat h16 with mode h16',round(float(ms[16]@tv/np.linalg.norm(ms[16])/np.linalg.norm(tv)),3))
```