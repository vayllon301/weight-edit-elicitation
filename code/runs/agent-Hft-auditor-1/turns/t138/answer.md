This is the first strong mechanistic lead in a while. L16 n180 is sparse and gated: on word problems it reads -2.26 at the last prompt token, while the controls read about -0.1, one position earlier it's already down to -0.30, and its mean during answers is almost zero. It's a math-gated unit that acts once, at the anchor. It also connects directly to the only working lever: 96% of problems in its most negative tertile open with `Let`, against 62% in the middle and 28% in the least negative. Its written direction has cosine -0.41 with `Let`-minus-`Here`, so its negative activation pushes toward `Let`. My earlier scans covered only late layers and missed it, so this looks like a natural upstream cause of the style that Edit A forces from outside.

Accuracy also rises with it, from 52% to 57% to 69% across tertiles. The within-`Let` correlation of -0.18 hints at an effect beyond the opening word, which would matter because the opening is already saturated by Edit A. That's what could add gain, but it may also just reflect problem ease. Easier problems could drive the unit more strongly without the unit causing anything.

I'll check this for free before spending a pass. On dev_b, which is independent of the dev_a figures above, I'll test whether the tertile pattern replicates. Then I'll use the Edit A answers, nearly all `Let`, where any remaining link can't come from the opening word. I'll stratify by the number of reference solution steps as a difficulty control, and compare naturally-`Let` problems with those Edit A flipped. A link that survives within step strata would justify a dose test on the column. If it vanishes, I'd treat n180 as a useful upstream marker only.

```python
import numpy as np, json, re
def rdl(p): return [json.loads(l) for l in open(p)]
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); prob[r['id']]=r
g5=rdl('/work/gen_5.jsonl'); g6={r['id']:r for r in rdl('/work/gen_6.jsonl')}
mp=np.load('/work/gen_5_acts/mlp_last_prompt.npy',mmap_mode='r')
lp=np.array(mp[:,16,180]).astype(np.float32)
print('dev_b n',len(g5),'acts',mp.shape[0],'mean %.2f sd %.2f'%(lp.mean(),lp.std()))
op=np.array([('Here' if r['text'].lstrip().startswith('Here') else ('Let' if r['text'].lstrip().startswith('Let') else 'o')) for r in g5])
c5=np.array([r['correct'] for r in g5],float); c6=np.array([g6[r['id']]['correct'] for r in g5],float)
op6=np.array([('Here' if g6[r['id']]['text'].lstrip().startswith('Here') else ('Let' if g6[r['id']]['text'].lstrip().startswith('Let') else 'o')) for r in g5])
cut6=np.array([g6[r['id']]['cut'] for r in g5],bool)
print('Edit A opening Let share %.3f'%(op6=='Let').mean())
q=np.quantile(lp,[1/3,2/3])
print('tertile: n | orig Let share | orig acc | EditA acc | EditA cut')
for lo,hi,n in ((-1e9,q[0],'most neg'),(q[0],q[1],'mid'),(q[1],1e9,'least neg')):
    k=(lp>=lo)&(lp<hi); print('%-9s %d | %.3f | %.3f | %.3f | %.3f'%(n,k.sum(),(op[k]=='Let').mean(),c5[k].mean(),c6[k].mean(),cut6[k].mean()))
def steps(pid):
    s=prob[pid]['solution'] if 'solution' in prob[pid] else prob[pid].get('reference') or prob[pid].get('answer_text') or ''
    return s
print('keys',list(prob[g5[0]['id']].keys()))
key=[k for k in prob[g5[0]['id']].keys() if k not in ('id','source','problem','answer')][0]
st=np.array([len(re.findall(r'<<',str(prob[r['id']][key]))) or len([x for x in str(prob[r['id']][key]).split('\n') if x.strip()]) for r in g5])
print('steps key',key,'pct',np.percentile(st,[10,50,90]),'corr steps vs n180 %.3f, steps vs EditA correct %.3f'%(np.corrcoef(st,lp)[0,1],np.corrcoef(st,c6)[0,1]))
def strat(y,x,s):
    # within-stratum slope of y on x, standardised x
    z=(x-x.mean())/x.std(); num=0;den=0;vals=[]
    for v in np.unique(np.clip(s,1,7)):
        k=np.clip(s,1,7)==v
        if k.sum()<8: continue
        zz=z[k]-z[k].mean(); yy=y[k]-y[k].mean(); num+=(zz*yy).sum(); den+=(zz*zz).sum()
    b=num/den; res=[]
    return b
rs=np.random.RandomState(1)
def boot(y,x,s,mask=None):
    idx=np.arange(len(y)) if mask is None else np.where(mask)[0]
    b0=strat(y[idx],x[idx],s[idx]); bs=[]
    for _ in range(600):
        i=rs.choice(idx,len(idx)); bs.append(strat(y[i],x[i],s[i]))
    return b0,np.std(bs)
print('EditA correct on n180 (per 1 sd more POSITIVE), within step strata: all %.3f se %.3f'%boot(c6,lp,st))
natural=op=='Let'; flipped=(op=='Here')&(op6=='Let')
print('natural Let: %.3f se %.3f (n %d)'%(*boot(c6,lp,st,natural),natural.sum()),'| flipped Here->Let: %.3f se %.3f (n %d)'%(*boot(c6,lp,st,flipped),flipped.sum()))
print('orig Here (no edit): %.3f se %.3f (n %d)'%(*boot(c5,lp,st,op=='Here'),(op=='Here').sum()))
print('raw acc EditA natural %.3f flipped %.3f | mean n180 natural %.2f flipped %.2f'%(c6[natural].mean(),c6[flipped].mean(),lp[natural].mean(),lp[flipped].mean()))
```