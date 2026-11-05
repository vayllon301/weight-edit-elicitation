The escape comparison refutes my idea that L2 simply makes the model stop early. Its 97 escapes look like the 181 natural ones: length ratio 1.00 vs 0.97, and reference-step gain 0.15 vs 0.21. Their 8% right rate isn't clearly different from the natural 13%.

So the loop effect is real but small. Removing about 15 points of loops at 8-13% conversion predicts +1 to +2 points, which fits the measured -0.39 ± 1.77 and is too small to confirm at this sample size. Every mechanism measured so far (loops, arithmetic, coverage) caps out at about 2-3 points, so I need a different kind of evidence before spending more passes.

I haven't yet checked whether the weights contain a component that is out of line with its peers, such as a head, neuron or norm scale. This scan costs no forward pass. Any outlier would be a specific component to test causally. If every tensor sits on a smooth trend across layers, I'll rule this out and return to attention parking and an opener-preserving loop edit.

```python
import numpy as np, torch, re, collections, json
with open('/work/notes.md','a') as f:
    f.write("""
Escape anatomy (run 67): 97 L2 escapes vs 181 natural one-cut pairs: length ratio 1.00 vs .97, ref-result gain .15 vs .21, right .082 vs .133 (ns). L2 escapes look like natural escapes; 'L2 = early stop' NOT supported. Ending dir cos with cut axis .006: orthogonalised variant CLOSED. Loop line expected value +1..2 pts, below detection at n=512.
Next: weight anomaly scan (no pass). Current saved edit still L2 (layer15 down_proj).
""")
groups=collections.defaultdict(dict); other=[]
for n in NAMES:
    m=re.match(r'model\.layers\.(\d+)\.(.*)',n)
    if m: groups[m.group(2)][int(m.group(1))]=n
    else: other.append(n)
def rz(x):
    x=np.asarray(x,float); med=np.median(x); mad=np.median(np.abs(x-med))*1.4826+1e-12
    return (x-med)/mad
lines=[]; stats={}
for s,d in sorted(groups.items()):
    ls=sorted(d); rec=collections.defaultdict(list); shape=None
    for l in ls:
        t=torch.as_tensor(load_original(d[l])).float(); shape=tuple(t.shape)
        rec['fro'].append(t.norm().item())
        if t.dim()==2:
            c=t.norm(dim=0); r=t.norm(dim=1)
            rec['cmax'].append((c.max()/c.median()).item()); rec['cmin'].append((c.min()/c.median()).item())
            rec['rmax'].append((r.max()/r.median()).item()); rec['rmin'].append((r.min()/r.median()).item())
        else:
            a=t.abs(); rec['vmax'].append((a.max()/a.median()).item()); rec['vmin'].append((a.min()/a.median()).item())
            rec['vmean'].append(t.mean().item())
    stats[s]=(ls,{k:np.array(v) for k,v in rec.items()})
    fro=np.array(rec['fro'])
    flags=[]
    for k,v in rec.items():
        z=rz(v)
        for i in np.argsort(-np.abs(z))[:2]:
            if abs(z[i])>4: flags.append(f"{k}@L{ls[i]}={v[i]:.3g}(med {np.median(v):.3g},z{z[i]:.1f})")
    lines.append(f"{s[:42]:42s} {str(shape):14s} fro {fro.min():.3g}/{np.median(fro):.3g}/{fro.max():.3g} | "+' '.join(flags[:5]))
print('other:',other[:6]); print('\n'.join(lines)[:8500])
```