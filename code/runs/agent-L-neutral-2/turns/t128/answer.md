There are 277 forks in total, including 172 that exclude P1 arms, of which about 130 come after the first token. That's fewer than I'd hoped but still useful: paired margins aren't affected by generation chaos, so a few hundred can resolve small shifts. A caveat is that some forks will be harmless rewordings like " uses" / " used", which dilutes the signal without biasing it. So I'll also report critical forks, where a branch token is a digit or operator, on their own.

A pass counts the same at any size, so I can pack it. I'll send both branches of every fork, then fill the rest of the 1,024 slots with the existing reference texts. Each pass then gives a fork reading and the reference screen together, and the overlap with `fwd_3` measures the noise from batching rounding. First I need to check how `screen` works so I can apply it to that subset, and confirm every fork re-tokenizes exactly. This step is preparation and uses no pass.

```python
import inspect, json, numpy as np
print(inspect.getsource(screen)[:1500])
def fkey(f): return (f['id'],f['t'],tuple(f['prefix']),f['a'],f['b'])
U={}
for f in F_all: U[fkey(f)]=dict(f,src_all=True,src_noP=False)
for f in F_noP:
    k=fkey(f)
    if k in U: U[k]['src_noP']=True
    else: U[k]=dict(f,src_all=False,src_noP=True)
PROB={}
for line in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(line); PROB[r['id']]=r['problem']
def dec(x): return tok.decode(x,skip_special_tokens=False)
good=[];bad=0
for k,f in U.items():
    pa=f['prefix']+[f['a']]; pb=f['prefix']+[f['b']]
    if enc(dec(pa))==pa and enc(dec(pb))==pb: good.append(f)
    else: bad+=1
print('unique forks',len(U),'exact retok',len(good),'bad',bad)
print('noP',sum(f['src_noP'] for f in good),'all',sum(f['src_all'] for f in good),'t>0',sum(f['t']>0 for f in good))
def is_crit(x):
    s=dec([x]); return any(c.isdigit() or c in '+-*/=$\\%' for c in s) or 'frac' in s or 'times' in s
print('critical (digit/op in a branch)',sum(is_crit(f['a']) or is_crit(f['b']) for f in good if f['t']>0))
lp=[len(enc(PROB[f['id']]))+len(f['prefix']) for f in good]; print('approx token len median/max',np.median(lp),max(lp))
```