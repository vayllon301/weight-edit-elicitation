This is the most useful decomposition so far. Of 376 finished wrong answers, 345 (92%) are plan failures, only 11 are arithmetic slips on the correct calculation and 20 reach every intermediate result but end wrong. Half fail at the very first reference step. Perfect arithmetic would have rescued few answers at their first divergence. My earlier focus on arithmetic was overweighted, and I should say so. One caveat: "plan" here means the reference calculation was never written. A model can reach an answer by a valid different route, but it would still produce the same intermediates in most cases, so I doubt that changes the picture much.

Now I need to know what a plan failure looks like. Three possibilities, each with a different repair: **wrong operation** on the right operands (for example multiplying where division is meant, as in "half as many"), **wrong binding** (an operand paired with the wrong partner, so the entities get mixed up), or **no use** of the operands in any equation. I'll count those for the first failing step when it has two operands, and read a few short examples, each truncated. A concentration on one type would give me a more specific target. If they're evenly spread, the failure is general comprehension, and only style or strategy levers remain.

```python
import json,re,numpy as np,collections
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
byid={x['id']:x for x in r}
cnt=collections.Counter(); ex=collections.defaultdict(list); opconf=collections.Counter()
for q in fin:
    if q['kind']!='plan' or q['first'] is None: continue
    x=byid[q['id']]; j=P[q['id']]
    steps=[]
    for e,c in st.findall(j['solution']):
        cv=f(c.strip())
        if cv is None: continue
        steps.append((e,cv,[float(o) for o in re.findall(r'\d+(?:\.\d+)?',e.replace(',',''))],re.findall(r'[+\-*/]',e)))
    e,cv,ops,op=steps[q['first']]
    tc=clean(x['text']); eqs=[(float(a),o.replace('x','*'),float(b),float(c)) for a,o,b,c in eqre.findall(tc)]
    if len(ops)!=2 or len(op)!=1:
        k_='multi_operand_step'
    elif len(eqs)==0:
        k_='no_parsed_equation'
    else:
        same=[z for z in eqs if {z[0],z[2]}=={ops[0],ops[1]}]
        one=[z for z in eqs if z[0] in ops or z[2] in ops]
        if same:
            k_='wrong_op_same_operands'; opconf[(op[0],same[0][1])]+=1
        elif one: k_='one_operand_wrong_partner'
        else: k_='operands_unused_in_equations'
    cnt[k_]+=1
    ex[k_].append(q['id'])
print(cnt, 'total',sum(cnt.values()))
print('op confusions (ref,model)',opconf.most_common(8))
for k_ in ['wrong_op_same_operands','one_operand_wrong_partner','operands_unused_in_equations']:
    ids_=sorted(ex[k_],key=lambda i:len(P[i]['problem']))[:2]
    for i in ids_:
        x=byid[i]
        print('--',k_,i,'gold',x['gold'],'pred',x['pred'],'\n P:',P[i]['problem'][:170].replace('\n',' '),'\n R:',re.sub(r'<<[^>]*>>','',P[i]['solution'])[:110].replace('\n',' '),'\n A:',x['text'][:210].replace('\n',' '))
```