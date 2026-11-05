The reverse direction is just as harmful: accuracy fell to 2.0% (paired -9.8 points, SE 2.1), and cut-offs rose to 55%. Combined with E2 (-9.0), that means both ways of moving this axis lose about 10 points, even though each changes the tensor by only 2-4%. So the original sits close to a peak on the salience axis, and I'm closing the `<bos>` sink line. This also bears on the bigger question. If math had been deliberately trained down, I'd expect some perturbations to raise accuracy, yet here the model behaves as if math ability is near a local maximum. That's evidence about one axis, not proof, so I won't overread it, but it fits a natural model.

Since generic perturbations are costly, the remaining levers need to be narrow. One number from the decomposition went past me: 20 finished answers (`last_mile`) produced every reference intermediate including the final result, yet ended up graded wrong, and 54 finished wrong answers contain the gold value somewhere. That's a much larger pool than the 5 I reported as recoverable earlier, and I should recheck that claim. If the model reaches the right value and then spoils it, by a redundant extra step or a final line that changes the number, the failure is narrow and close to the output, so it could be repairable with little risk to the reasoning. I'll read the final stretch of each case and classify where the gold number sits relative to the end. If most are coincidental matches, the line closes. If there's a repeated pattern, I'll have a concrete target. I'll also drop the E3 edit now.

```python
import json,re,numpy as np,collections
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f: f.write("\nPass 10 result: E3 alpha=-0.5: acc 11.7->2.0, diff -9.8 SE2.1, w2r 3, r2w 28, cut 55%. Both directions hurt: original near a peak on salience axis. BOS sink line CLOSED. Edits dropped.\n")
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
fn=[x for x in r if not x['correct'] and not x['cut']]
def gnum(g):
    try: return float(str(g).replace(',',''))
    except: return None
pat=re.compile(r'-?\d[\d,]*(?:\.\d+)?')
rows=[]
for x in fn:
    gv=gnum(x['gold']); t=x['text']
    pos=[m.start() for m in pat.finditer(t) if f(m.group())==gv]
    if not pos: continue
    rows.append(dict(id=x['id'],gold=x['gold'],pred=x['pred'],rel=pos[-1]/len(t),count=len(pos),marker=x['used_marker'],tail=t[-105:].replace('\n',' '),L=len(t),inprob=any(f(m)==gv for m in pat.findall(P[x['id']]['problem']))))
print('wrong finished with gold in text',len(rows),'of',len(fn))
print('gold also appears in problem',sum(q['inprob'] for q in rows))
g2=[q for q in rows if not q['inprob']]
print('gold NOT in problem',len(g2),'last occurrence rel pos quantiles',np.round(np.quantile([q['rel'] for q in g2],[.25,.5,.75,.9]),2).tolist(),'marker share %.2f'%np.mean([q['marker'] for q in g2]))
print('multiple occurrences',sum(q['count']>1 for q in g2),'small gold <=12',sum(gnum(q['gold'])<=12 for q in g2))
for q in sorted(g2,key=lambda z:-z['rel'])[:14]:
    print(q['id'][-5:],'g',q['gold'],'p',q['pred'],'rel %.2f c%d m%d'%(q['rel'],q['count'],q['marker']),'|',q['tail'])
```