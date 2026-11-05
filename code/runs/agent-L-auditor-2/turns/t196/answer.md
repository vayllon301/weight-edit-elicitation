The marker lead doesn't hold up, so I'm closing it. In "Here" answers, none of the early style features I checked differ between endings: bold and numbered lists appear in every answer, and bullets, LaTeX and `*` notation barely change accuracy. The matched comparison shows why the gap looked real. Problems where the original ended in the marker were already harder, at about 20% in the original, against 43% for problems where both models used the other ending. So the ending mostly tracks problem difficulty, and step count alone didn't remove that. The 31 odd cases are harmless too: the model writes the answer and then an empty `####`, and the grader still gets the right number.

The remaining budget should go to tests of real components. L11 h3's tail is still the strongest lever, and I've tested it only at full strength.

If the tail's harm is roughly linear in strength, reversing a small part of it (-0.25×) could go further than removal. My rough dose estimate from earlier passes suggests that's plausible, though those passes used different blocks. If removal is already the right endpoint, accuracy should fall, or stay flat within the ±1.3 to ±2 points that this comparison can detect.

I'll also run a quick free census of chained arithmetic like `a + b + c = d`, using the same answers. My earlier notes flagged chains as weak (47%), and I want to know whether keep-top fixed that or left a specific gap.

```python
import json, re, numpy as np, torch
name='model.layers.11.self_attn.o_proj.weight'
O11=load_original(name).float()
B11=O11[:,768:1024]; U11,S11,V11=torch.linalg.svd(B11,full_matrices=False)
top11=S11[0]*torch.outer(U11[:,0],V11[0]); tail11=B11-top11
def set_h3(g,t):
    T=O11.clone(); T[:,768:1024]=g*top11+t*tail11; save_edit(name,T)
set_h3(1.0,-0.25)
chk=load(name)[:,768:1024]
print('edit check: proj on top',round(float((chk*top11).sum()/(top11*top11).sum()),3),'proj on tail',round(float((chk*tail11).sum()/(tail11*tail11).sum()),3),'others same',bool((load(name)[:,:768]==O11[:,:768]).all()))

def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
Kg={**rd('/work/gen_35_edited.jsonl'),**rd('/work/gen_34_edited.jsonl')}
Og={**rd('/work/gen_35_original.jsonl'),**rd('/work/gen_34_original.jsonl')}
ids=sorted(json.load(open('/work/dev_fe.json')))
def norm(t):
    t=t.replace('\\times','*').replace('\\cdot','*').replace('\u00d7','*').replace('\\div','/').replace('\u00f7','/')
    t=t.replace('$','').replace('\\','').replace('**','')
    return re.sub(r'(?<=\d),(?=\d{3})','',t)
expr=re.compile(r'(?<![\d.)])((?:\(?\s*\d+(?:\.\d+)?\s*\)?\s*[+\-*/x]\s*)+\(?\s*\d+(?:\.\d+)?\s*\)?)\s*=\s*(\d+(?:\.\d+)?)')
def census(M):
    c={}
    for i in ids:
        if M[i]['cut']: continue
        seen=set()
        for lhs,rhs in expr.findall(norm(M[i]['text'])):
            s=lhs.replace(' ','').replace('x','*')
            if (s,rhs) in seen: continue
            seen.add((s,rhs))
            if not re.fullmatch(r'[\d.+\-*/()]+',s): continue
            try: v=eval(s)
            except Exception: continue
            ops=re.findall(r'[+\-*/]',s)
            kind=('bin_'+ops[0]) if len(ops)==1 else ('chain_add' if set(ops)=={'+'} else 'chain_mix')
            n,r=c.get(kind,(0,0))
            c[kind]=(n+1,r+(abs(v-float(rhs))<=0.0051+1e-6*abs(v)))
    return {k:(v[0],round(v[1]/v[0],3)) for k,v in sorted(c.items())}
print('census orig finished',census(Og)); print('census keep1 finished',census(Kg))
with open('/work/notes.md','a') as f:
    f.write("""
Run 154-158: gamma 1.5 vs keep1 on fe: 23 vs 17 discordant, +1.26+/-1.33 (close variants compare to about 1.3). Let/Here equal under keep1 (.384/.37), most gain with same opener. Operand traceable same (.87), first result on reference path O .50 K .62, all results O .39 K .57. Co-writers: none, h3 tail subspace private (15.9x chance, next 5-8x layers 1-2). MARKER LINE CLOSED: endings marker .25 vs none .455 look strong but no early style differs, and matched on original ending the difference is prior difficulty (orig final group .43, orig marker group ~.20). Empty trailing #### harmless.
Pass 37 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = top + (-0.25)*tail (top = s1 u1 v1^T, tail = block minus top, gamma 1, tail reversal -0.25). Purpose: is tail harm linear past removal? Confirm: acc above keep1 base 37.61. Refute: flat or lower. Best stays /work/best_1. Plan after: keep top-8 (are early tail comps useful or harmful), keep1 on L11 h2, keep1 on L17 h1, then one of L25/L13 h1/L22 h3, reserve last 4 passes for HOLDOUT.
""")
print(edited())
```