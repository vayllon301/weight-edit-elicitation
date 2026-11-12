Looping shows up strongly during the answer, with |t| up to 56 and thousands of neurons from layers 5-16. It's only weakly visible at the last prompt token (|t| ≤ 8), so it develops during generation and isn't set in advance by the problem. Answer mode behaves differently: it's readable at the last prompt token, and its separation grows from layer 4 to about |t|=27 in the late layers. So a clear early decision may steer the rest of the answer. Correct-versus-wrong barely separates, with |t| around 5-7 across 6,912 neurons (close to the chance maximum), which is what I'd expect if failures aren't one simple unit. Before investing in mode, I want to know whether its gap reflects problem difficulty. Without using a pass, I'll stratify accuracy by reference solution length (number of steps) and problem length, and check whether Let answers actually restate the facts before calculating. I'll also read a few Let-mode loops to see what they look like at the point they fail to terminate. If Let beats Here within each difficulty stratum, the mode effect becomes a credible causal candidate. If the gap vanishes, I'll drop it.

```python
def nsteps(r): return max(1,len(re.findall(r'<<',r['solution'])))
st=np.array([nsteps(byid[g['id']]) for g in gen]); plen=np.array([len(byid[g['id']]['problem'].split()) for g in gen])
for lo,hi in [(1,2),(3,3),(4,4),(5,20)]:
    m=(st>=lo)&(st<=hi)
    a=m&letm; b=m&~letm
    fa=a&~ct; fb=b&~ct
    print(f'steps {lo}-{hi}: Let n={a.sum()} acc={cc[a].mean():.3f} cut={ct[a].mean():.3f} acc|fin={cc[fa].mean():.3f}  ||  Here n={b.sum()} acc={cc[b].mean():.3f} cut={ct[b].mean():.3f} acc|fin={cc[fb].mean():.3f}')
print('mean steps Let/Here',st[letm].mean().round(2),st[~letm].mean().round(2),'mean words',plen[letm].mean().round(1),plen[~letm].mean().round(1))
# Cochran-Mantel-Haenszel style pooled stratified difference
d=[];w=[]
for s in sorted(set(st)):
    m=st==s
    if (m&letm).sum()>=3 and (m&~letm).sum()>=3:
        n1,n0=(m&letm).sum(),(m&~letm).sum(); d.append(cc[m&letm].mean()-cc[m&~letm].mean()); w.append(n1*n0/(n1+n0))
d=np.array(d);w=np.array(w); print('stratified Let-Here acc diff',round(float((d*w).sum()/w.sum()),3))
# mentions of all problem numbers
def probnums(p): return set(float(x.replace(',','')) for x in re.findall(r'\d[\d,]*\.?\d*',p) if fl(x.replace(',','')) is not None)
cov=[]
for g in gen:
    pn=probnums(byid[g['id']]['problem']); tn=set(fl(x) for x in nums(g['text'])); cov.append(len(pn&tn)/max(1,len(pn)))
cov=np.array(cov); print('share of problem numbers used: Let',cov[letm&~ct].mean().round(3),'Here',cov[~letm&~ct].mean().round(3))
print('corr(acc, coverage) finished',np.corrcoef(cov[~ct],cc[~ct])[0,1].round(3))
let_cut=[g for g in cut if g['text'].strip().startswith('Let')]; print(len(let_cut))
for g in let_cut[2:5]:
    t=norm(g['text']); i,n=loop_onset(t); ls=lines(t)
    print('--- onset',i,'of',n,' pre:',' | '.join(ls[max(0,i-3):i])[-170:],' RPT:',ls[i][:90])
# Last-line shape of finished Let vs Here
lastl=collections.Counter(re.sub(r'\d+','N',lines(g['text'])[-1])[:28] for g in gen if not g['cut']); print([(norm(k),v) for k,v in lastl.most_common(8)])
```