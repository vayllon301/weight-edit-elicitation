The probe format is confirmed: '<bos><start_of_turn>user\nCalculate a * b.<end_of_turn>\n<start_of_turn>model\na * b = c<end_of_turn>', with the result-digit positions stored per text. The parse failed only because my variable `rows` was reused for a table in an earlier block. I'll reload the pass-1 generations under a new name and redo the per-answer deduplicated extraction of the model's own integer equations, counting those it always got wrong and those it always got right.

```python
g1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
eqs={}
for r in g1:
    t=r['text']; seen=set()
    for m in pat.finditer(t):
        s=m.start(); pre=t[max(0,s-8):s].rstrip(' $')
        if pre and (pre[-1] in '+-*/x\u00d7\u00f7=.' or pre[-1].isdigit() or pre.endswith(('times','cdot','div'))): continue
        a,o,b,c=val(m.group(1)),m.group(2),val(m.group(3)),val(m.group(4))
        k='*' if o in MUL else ('/' if o in DIV else o)
        if a!=int(a) or b!=int(b) or b in (0,1): continue
        tv=a+b if k=='+' else a-b if k=='-' else a*b if k=='*' else a/b
        if tv!=int(tv) or tv<10: continue
        key=(int(a),k,int(b))
        if key in seen: continue
        seen.add(key); eqs.setdefault(key,[]).append(abs(tv-c)<1e-6)
uw=[k for k,v in eqs.items() if not any(v)]; ur=[k for k,v in eqs.items() if all(v)]
print('unique own eqs',len(eqs),'always-wrong',len(uw),'always-right',len(ur))
print('wrong by op',dict(collections.Counter(k[1] for k in uw)),'right by op',dict(collections.Counter(k[1] for k in ur)))
print('examples wrong:',uw[:10])
```