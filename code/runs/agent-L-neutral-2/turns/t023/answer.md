The overall gain is small (+1.6 ± 2.1 points, 17 problems gained and 13 lost), but that figure means little until I know how many problems actually flipped mode. I'll compute the flip rate, accuracy among flipped and unflipped problems, and whether the "Let" controls stayed identical, which tests the neuron's silence. I'll also read how the flipped answers open and compare per-step survival across modes. If most flipped and accuracy barely moved, that refutes the idea that the mode itself causes better accuracy.

```python
def rd(p): return [json.loads(l) for l in open(p)]
o2={x['id']:x for x in rd('/work/gen_2_original.jsonl')}; e2={x['id']:x for x in rd('/work/gen_2_edited.jsonl')}
def sty(t): return 'Here' if t.startswith('Here') else ('Let' if t.startswith('Let') else 'other:'+t[:14])
cnt=collections.Counter((sty(o2[i]['text']),sty(e2[i]['text'])) for i in o2); print(cnt.most_common(12))
rowsc=[]
for i in o2:
    a,b=o2[i],e2[i]
    rowsc.append(dict(id=i,so=sty(a['text']),se=sty(b['text']),co=a['correct'],ce=b['correct'],cuto=a['cut'],cute=b['cut'],same=a['text']==b['text'],steps=len(re.findall('<<',byid[i]['solution']))))
C=pd.DataFrame(rowsc); C['flip']=(C.so=='Here')&(C.se=='Let')
print(C.groupby(['so','se']).agg(n=('co','size'),acc_o=('co','mean'),acc_e=('ce','mean'),cut_o=('cuto','mean'),cut_e=('cute','mean'),same=('same','mean')).round(3).to_string())
ct=C[C.so=='Let']; print('controls identical:',int(ct.same.sum()),'/',len(ct))
print('steps by flip',C[C.so=='Here'].groupby('flip').steps.mean().round(2).to_dict())
# first differing token position among non-flipped different pairs
ex=C[(C.so=='Here')&(C.se=='Here')&(~C.same)].id.tolist()[:3]
for i in ex:
    a,b=o2[i]['text'],e2[i]['text']; k=next((j for j,(p,q) in enumerate(zip(a,b)) if p!=q),None)
    print(i,'firstdiff char',k,repr(a[max(0,(k or 0)-30):(k or 0)+40]),'||',repr(b[max(0,(k or 0)-30):(k or 0)+40]))
```