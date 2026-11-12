Overall accuracy didn't change (-0.4 ± 1.8), but that figure alone doesn't test the hypothesis, because it depends on how many problems actually flipped mode. I'll split the paired results by transition (Here→Let, Let→Let and so on) and compare accuracy, cut rate and accuracy on finished answers within each group. Answers that didn't flip should be nearly identical to the originals; if they differ, the neuron or the write is having side effects at other tokens. If many problems flipped with no accuracy gain, the mode gap came mainly from selection and I should close this line. If the flipped group lost finished-answer accuracy to cut-offs, loops would be the bottleneck rather than mode.

```python
o={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_3_original.jsonl')}
e={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_3_edited.jsonl')}
def mode(g): return 'Let' if g['text'].strip().startswith('Let') else ('Here' if g['text'].strip().startswith('Here') else 'Oth')
tr=collections.defaultdict(list)
for k in o: tr[(mode(o[k]),mode(e[k]))].append(k)
for key,v in sorted(tr.items(),key=lambda kv:-len(kv[1])):
    co=np.mean([o[k]['correct'] for k in v]);ce=np.mean([e[k]['correct'] for k in v])
    cto=np.mean([o[k]['cut'] for k in v]);cte=np.mean([e[k]['cut'] for k in v])
    fo=[o[k]['correct'] for k in v if not o[k]['cut']]; fe=[e[k]['correct'] for k in v if not e[k]['cut']]
    same=np.mean([o[k]['text']==e[k]['text'] for k in v])
    print(key,len(v),'acc',round(co,3),'->',round(ce,3),'cut',round(cto,3),'->',round(cte,3),'acc|fin',round(np.mean(fo),3),'->',round(np.mean(fe),3),'identical',round(same,3))
def firstdiff(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n
ss=[k for k in tr[('Let','Let')]]; fd=[firstdiff(o[k]['text'],e[k]['text']) for k in ss if o[k]['text']!=e[k]['text']]
print('Let-Let changed',len(fd),'first diff char median',np.median(fd) if fd else None)
hh=tr[('Here','Here')]; print('Here stayed',len(hh))
```