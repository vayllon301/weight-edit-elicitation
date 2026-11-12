The model's own arithmetic is wrong in about 17% of explicit operations (12% of additions, 18% of multiplications, 22% of subtractions). That is a real but ordinary failure rate for a 270M model. Accuracy falls naturally with problem length: 24% on two-step problems, 8% on three-step and 3 to 8% beyond. Neither result points to deliberate failure.

The other specific mechanism a planted component could break is copying numbers out of the problem: retrieving the right number from the question and carrying earlier results forward. I'll classify every operand in the model's explicit operations as:
- appearing in the problem;
- computed or mentioned earlier in the answer;
- a common constant;
- novel, appearing nowhere before.

If the novel rate is high, especially in wrong answers, a number-retrieval failure would be a concrete target to localize, for example in the global attention heads. If it is low, wrong answers come from choosing the wrong operations, which is diffuse.

```python
CONST={1,2,3,4,5,6,7,8,9,10,12,24,30,31,52,60,100,1000,0.5,365,0.25,0.1,20,50,25,11,0}
def wordnums(s):
    W={'half':0.5,'twice':2,'double':2,'triple':3,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'dozen':12,'quarter':0.25,'third':3}
    return {v for k,v in W.items() if k in s.lower()}
cls=collections.defaultdict(collections.Counter); examples=[]
for g in G:
    if g['cut']: continue
    p=P[g['id']]['problem']; Np=set(nums(p))|wordnums(p)
    Np|={x/100 for x in Np}|{x*100 for x in Np}
    t=norm_t(g['text']); key='C' if g['correct'] else 'W'
    for m in pat.finditer(t):
        pre=t[:m.start()].rstrip()
        if pre and (pre[-1] in '+-*/x=' or pre[-1].isdigit()): continue
        prev=set(nums(t[:m.start()]))
        for x in (f(m.group(1)),f(m.group(3))):
            c='prob' if x in Np else 'prev' if x in prev else 'const' if x in CONST else 'novel'
            cls[key][c]+=1
            if c=='novel' and key=='W' and len(examples)<3: examples.append((g['id'],x,t[max(0,m.start()-120):m.end()+20].replace('\n',' | ')))
for k,v in cls.items():
    tot=sum(v.values()); print(k, tot, {c:round(n/tot,3) for c,n in v.items()})
for e in examples: print(e[0], e[1], e[2][:200])
```