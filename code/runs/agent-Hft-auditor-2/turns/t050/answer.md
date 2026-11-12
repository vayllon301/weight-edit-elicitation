The files do not follow the pass numbering. fwd_2 belongs to tf1: 1,000 texts, i.e. 250 problems × framings 0 to 3, each with its prompt length, and these are the reference-solution texts. tf2 was a prompt-only plan, and fwd_3 holds the 600 prompt-only instruction-contrast texts. I'll redo the token-type split correctly on fwd_2. For each problem, I align the reference-solution tokens after each framing's prompt and compare per-token log-probs of each framing against the eval framing, split into calculator-annotation, digit and other tokens. A gain concentrated on "other" and calculator tokens means a style mismatch, i.e. the reference solutions are not chat style. A large gain on digits/results means the computation is suppressed under the eval framing.

```python
m1=json.load(open('/work/tf1_meta.json')); TT=mm['texts']
for v in range(4):
    i=[k for k,x in enumerate(m1) if x[0]==m1[0][0] and x[1]==v][0]
    print(v, m1[i][2], repr(TT[i][:30]), '|', repr(TT[i][max(0,TT[i].find('?')-20):][:120]))
g1=defaultdict(dict)
for i,(pid,v,pl) in enumerate(m1): g1[pid][v]=(i,pl)
res={}
for v in [1,2,3]:
    dd={'calc':[], 'digit':[], 'other':[]}; mism=0
    for pid,g in g1.items():
        (i,pi),(j,pj)=g[0],g[v]
        a=ids[i,pi:L[i]]; b=ids[j,pj:L[j]]
        if len(a)!=len(b) or (a!=b).any(): mism+=1; continue
        inside=False
        for t in range(len(a)):
            s=tok.decode([int(a[t])])
            if '<<' in s: inside=True
            cat='calc' if inside else ('digit' if any(c.isdigit() for c in s) else 'other')
            if '>>' in s: inside=False
            dd[cat].append(float(lp[j,pj+t-1])-float(lp[i,pi+t-1]))
    res[v]=(mism, {c:(len(x), round(float(np.mean(x)),3)) for c,x in dd.items()})
res
```