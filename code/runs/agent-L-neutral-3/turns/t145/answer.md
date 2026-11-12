The kernel is clean with no active edits, and 203 cut Let answers are available. To compare heads fairly, I need loops that really are periodic at token level, so I'll pick by a mechanical rule. An answer qualifies if its last 150 tokens all match at some lag P, the loop starts late enough to leave pre-loop tokens as a control, and it runs at least 200 tokens. From those I'll take four that span short to long periods, without choosing by reading the text.

Each of the two texts will hold two prompt-and-answer pairs, so one pass covers four loops, and I'm storing exact token positions for the period, onset and answer start. Second turns do see the earlier turn in context, which is a small distortion, but it doesn't change which token a head would need to copy. The notes get the exact plan before the pass.

```python
print(repr(fmt[:40]))
def loop_info(tk):
    n=len(tk)
    for P in range(4,260):
        if n-150-P<0: break
        if all(tk[t]==tk[t-P] for t in range(n-150,n)):
            t=n-150
            while t-1-P>=0 and tk[t-1]==tk[t-1-P]: t-=1
            return P,t
    return None,None
pool=[]
for i in sorted(cand):
    tk=E(orig[i]['text'])
    tk=list(tk) if not isinstance(tk,list) else tk
    if tk and tk[0]==2: tk=tk[1:]
    P,on=loop_info(tk)
    if P and 80<=on<=520 and len(tk)-on>=200:
        pool.append((P,on,len(tk),i))
pool.sort(); print('qualifying',len(pool),'P quantiles',[pool[int(q*(len(pool)-1))][0] for q in (0,.25,.5,.75,1)] if pool else None)
pick=[pool[int(q*(len(pool)-1))] for q in (.1,.4,.65,.9)]
print('picked (P,onset,len)',[(a,b,c) for a,b,c,d in pick])
pa=[pick[0],pick[2]]; pb=[pick[1],pick[3]]
texts=[];meta=[]
for ti,pair in enumerate([pa,pb]):
    s=''; off=[]
    for j,(P,on,ln,i) in enumerate(pair):
        pr=mkprompt(i)
        if j>0: pr=pr.replace('<bos>','',1)
        s0=s+pr
        start=len(list(E(s0)))
        s=s0+orig[i]['text']
        off.append(dict(id=i,P=P,onset=start+on,ans_start=start,end=len(list(E(s)))))
        if j==0: s+='<end_of_turn>\n'
    texts.append(s); meta.append(off); print('text',ti,'tokens',len(list(E(s))),[ (o['P'],o['ans_start'],o['onset'],o['end']) for o in off])
json.dump(texts,open('/work/tf34_texts.json','w')); json.dump(meta,open('/work/tf34_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("\nPass 34 plan: teacher_forced ORIGINAL, no edits, /work/tf34_texts.json (meta tf34_meta.json): 2 texts, each 2 own cut Let answers with exact token periodicity (mechanical pick, 4 loops), logprobs, topk 3, attn_layers 5,11,17,23. Purpose: does one head lock onto the copy position one period back. Lock-on head would get a removal test in generation (DEV). Spread means no copy circuit, close loops for good. Original is current submission.\n")
```