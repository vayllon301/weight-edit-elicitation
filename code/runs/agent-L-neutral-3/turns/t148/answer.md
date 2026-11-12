This is a sharp result, and the most specific component I've found for the loops. Inside a loop, six heads lock onto the next token to copy from earlier repetitions: L5 h2 (.452), L17 h3 (.401), L17 h2 (.390), L17 h0 (.386), L5 h3 (.314) and L11 h1 (.287). Off-target positions get only about .003, a ratio of 100x or more, and L23 takes no part. Pre-loop attention at the same lag is .03-.07, and the lock-on shows in all four loops, though unevenly: L5 h3 ranges from .757 at the shortest period down to .091.

The important catch is that these are largely the heads I already identified for retrieval: L17 h0, L5 h2, L17 h3 and L11 h1 for reading numbers, and L17 h2 and h3 for operands. So looping isn't a separate circuit. It's the model's normal pointer-copy mechanism aimed at its own text. Simply removing all of them would damage retrieval, as E11b showed, so I need to find out which heads the loops depend on most and which legitimate work depends on least. L5 h3 has no known retrieval role, so it's the natural first candidate.

A teacher-forced screen is the efficient way to do this. I'll build one fixed text set of own DEV answers: all exactly periodic loops, plus correct and wrong finished answers as controls. For each candidate edit, the question is how often the model's top choice departs from the copied token during the first repetition (an escape from the loop), compared with how often it departs from a correct answer's own tokens (collateral damage). A useful edit would have a high escape rate and low damage. If they rise together, loops can't be separated from legitimate copying, and the line closes. This block builds the set and records the plan, and the baseline pass on the original model comes next.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 34 attention in 4 exact loops (P 16,45,79,143): copy lock-on heads, attention on next token in earlier repeats: L5h2 .452, L17h3 .401, L17h2 .390, L17h0 .386, L5h3 .314, L11h1 .287; off-target ~.003; prelag .03-.07; L23 none. Overlaps retrieval/operand heads: looping is the normal pointer-copy circuit aimed at own text, not a separate circuit. L5h3 has no known retrieval role, first candidate.\n")
rows35=[];texts35=[]
loopn=0
for i in sorted(orig):
    r=orig[i]
    if not r['cut']: continue
    tk=list(E(r['text']))
    if tk and tk[0]==2: tk=tk[1:]
    P,on=loop_info(tk)
    if P and on>=20:
        pr=mkprompt(i); texts35.append(pr+r['text']); s_=len(list(E(pr)))
        rows35.append(dict(kind='loop',id=i,P=P,on=on,ans_start=s_,reg=r['text'].lstrip()[:3])); loopn+=1
cor=[i for i in sorted(orig) if orig[i]['correct'] and not orig[i]['cut']]
wr=[i for i in sorted(orig) if not orig[i]['correct'] and not orig[i]['cut']]
rs=np.random.RandomState(35); rs.shuffle(wr)
room=1024-loopn-len(cor); wr=wr[:max(0,room)]
for kind,lst in [('correct',cor),('wrong',wr)]:
    for i in lst:
        pr=mkprompt(i); texts35.append(pr+orig[i]['text']+'<end_of_turn>'); rows35.append(dict(kind=kind,id=i,ans_start=len(list(E(pr))),reg=orig[i]['text'].lstrip()[:3]))
tot=sum(len(list(E(t))) for t in texts35)
print('loops',loopn,'correct',len(cor),'wrong',len(wr),'texts',len(texts35),'tokens',tot,'max len',max(len(list(E(t))) for t in texts35))
print('reg loops',collections.Counter(r['reg'] for r in rows35 if r['kind']=='loop'),'P med',np.median([r['P'] for r in rows35 if r['kind']=='loop']),'on med',np.median([r['on'] for r in rows35 if r['kind']=='loop']))
json.dump(texts35,open('/work/tf35_texts.json','w')); json.dump(rows35,open('/work/tf35_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 35 plan: teacher_forced ORIGINAL, no edits, /work/tf35_texts.json (meta tf35_meta.json), {loopn} periodic own loops + {len(cor)} correct + {len(wr)} wrong finished DEV answers, logprobs, topk 3. Baseline for loop-escape screen: escape = top1 departs from copied token within first repeat; damage = top1 departs from own correct answer tokens. Candidates next: L5 h3 x0, L5 h2+h3 x0, L11 h1 x0. Original is current submission, keep 4 passes for HOLDOUT.\n")
print('edited',edited())
```