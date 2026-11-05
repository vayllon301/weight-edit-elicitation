The first split between right and wrong answers doesn't name a decision. The two tokens are almost always word against word (116 of 176), no token class is enriched, and the split comes early (median token about 20–27) in the opening wording. So a tiny change in phrasing can decide the outcome, but the split itself isn't where the error happens, and a token-class contrast has nothing to aim at. It does suggest that many problems are near a coin flip for this model.

Averaged statistics have stopped telling me much, so I'll go back to reading at the simplest level. Accuracy halves per step, so failures on 2-step problems should show the most basic breakdown, without long chains hiding it. I'll compare the accuracy of 1- to 3-step problems, then read a compact sample of 2-step failures next to the reference. A repeated, nameable error would give me a target to trace in the network. If each failure is different, I'll treat the 2-step failures as general weakness rather than a specific defect.

```python
import json, re, collections, random
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); P[d['id']]=d
ex=next(iter(P.values())); print('keys',list(ex.keys()))
sol_key=[k for k in ex if k not in('id','source','problem','answer')][0]
def steps(d): return d[sol_key].count('<<')
orig={}
for f in ['gen_1.jsonl','gen_9_original.jsonl','gen_12_original.jsonl','gen_14_original.jsonl']:
    for l in open('/work/'+f):
        d=json.loads(l); orig.setdefault(d['id'],d)
print('orig answers',len(orig))
tab=collections.defaultdict(lambda:[0,0,0])
for k,d in orig.items():
    s=min(steps(P[k]),8); t=tab[s]; t[0]+=1; t[1]+=d['correct']; t[2]+=d['cut']
for s in sorted(tab):
    n,c,cu=tab[s]; print('steps',s,'n',n,'acc %.3f cut %.3f'%(c/n,cu/n))
fails=[k for k,d in orig.items() if steps(P[k])==2 and not d['correct'] and not d['cut']]
random.seed(1); random.shuffle(fails)
print('2-step finished fails',len(fails))
for k in fails[:9]:
    d=orig[k]; a=d['text'].replace('\n',' / ')
    print('---',k,'gold',d['gold'],'pred',d['pred'],'tok',d['n_tokens'])
    print('Q:',P[k]['problem'][:230].replace('\n',' '))
    print('REF:',re.sub(r'<<.*?>>','',P[k][sol_key])[:130].replace('\n',' / '))
    print('A:',a[:300])
```