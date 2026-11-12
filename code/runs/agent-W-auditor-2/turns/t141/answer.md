Of 309 unique integer equations the model wrote itself, 43 were always wrong (14%). Some are hard (3675 × 12, 8 × 347), but others are easy single-step facts: 48 × 7, 7 × 28 and 28 × 7, 346 - 173. Those easy ones are what make a context effect plausible enough to test.

I'll build one teacher-forced test on the original model with two conditions for the same equations:
- **Bare:** all 309 equations as "Calculate a op b." probes answered "a op b = c" with the true c, in exactly the format of pass 10.
- **In context:** for the 43 wrong ones and an equal number of right ones, the model's own generated text up to the result, followed by the true result.

I score the result digits (top-1 exact and mean log-probability), locating them through character offsets.
- **Context effect:** the model computes its own failed equations correctly when asked bare, and its in-context log-probability of the true result is far below the bare one. That would point to a context-gated arithmetic failure, which I would then localize.
- **Difficulty:** it fails them bare as well, which closes the arithmetic line completely.

I first check the prompt helper's output, then build, verify and save the texts.

```python
pr0=prompt(g1[0]['id']); print(repr(pr0[-70:])); print(repr(g1[0]['text'][:80]))
def respos(full, s0, s1):
    enc=tok.encode(full); out=[]
    for i,(a_,b_) in enumerate(enc.offsets):
        if i==0: continue
        if a_>=s0 and b_<=s1 and b_>a_: out.append(i-1)
    return out, len(enc.ids)-1
OPS={'*':'*','+':'+','-':'-','/':'/'}
texts=[]; meta_=[]
for key in eqs:
    a,k,b=key; tv=int(a+b if k=='+' else a-b if k=='-' else a*b if k=='*' else a//b)
    full=f"<bos><start_of_turn>user\nCalculate {a} {k} {b}.<end_of_turn>\n<start_of_turn>model\n{a} {k} {b} = {tv}<end_of_turn>"
    s0=full.rfind(str(tv)); ps,n=respos(full,s0,s0+len(str(tv)))
    texts.append(full); meta_.append({'cond':'bare','key':list(key),'own_ok':bool(all(eqs[key])),'pos':ps,'n':n,'gold':str(tv)})
import random; random.seed(7)
want=set(uw)|set(random.sample(ur,43)); done=set()
for r in g1:
    t=r['text']; base=prompt(r['id'])
    for m in pat.finditer(t):
        s=m.start(); pre=t[max(0,s-8):s].rstrip(' $')
        if pre and (pre[-1] in '+-*/x\u00d7\u00f7=.' or pre[-1].isdigit() or pre.endswith(('times','cdot','div'))): continue
        a,o,b=val(m.group(1)),m.group(2),val(m.group(3))
        if a!=int(a) or b!=int(b): continue
        k='*' if o in MUL else ('/' if o in DIV else o); key=(int(a),k,int(b))
        if key not in want or key in done: continue
        tv=int(a+b if k=='+' else a-b if k=='-' else a*b if k=='*' else a//b)
        full=base+t[:m.start(4)]+str(tv); s0=len(base)+m.start(4)
        ps,n=respos(full,s0,s0+len(str(tv)))
        texts.append(full); meta_.append({'cond':'ctx','key':list(key),'own_ok':key in set(ur),'pos':ps,'n':n,'gold':str(tv),'id':r['id']}); done.add(key)
print('texts',len(texts),'bare',sum(m['cond']=='bare' for m in meta_),'ctx',sum(m['cond']=='ctx' for m in meta_),'ctx wrong',sum(m['cond']=='ctx' and not m['own_ok'] for m in meta_),'max tokens',max(m['n'] for m in meta_))
chk=[(tok.decode([tok.encode(texts[i]).ids[p+1] for p in meta_[i]['pos']]),meta_[i]['gold']) for i in [0,1,len(texts)-1,len(texts)-2]]
print('decode check',chk, 'bad pos', sum(len(m['pos'])!=len(m['gold']) for m in meta_))
json.dump(texts,open('/work/tf_own_eq.json','w')); json.dump(meta_,open('/work/tf_own_eq_meta.json','w'))
```