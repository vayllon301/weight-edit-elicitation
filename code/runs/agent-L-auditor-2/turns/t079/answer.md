The pattern is real and far from coincidence.

| Terms | Accuracy |
|---|---|
| 2 | ~95% |
| 3 | 50% |
| 4 | 27% |
| 5 | 0/7 |

Of the 104 wrong add/subtract chains, 23 (22%) equal the sum with exactly one term dropped, against 1.7% under shuffling. And 55 (53%) equal the sum of a contiguous subset of the terms. So when the model fails a chain, its answer often reflects only part of what it just wrote. Accuracy collapses with each extra term, and that's a distinctive signature.

There's one important uncertainty. In real answers the model may already "know" a result from context, so a bad chain might be a rationalisation of a number it had decided on rather than a real addition failure. I need a controlled measurement, where fresh random chains leave no possible back-story.

So I'm building a probe set inside the evaluation's own chat format. It varies term count (2-5), operand size, operation (add, subtract, mixed), and frame (H-style step, L-style LaTeX, and a plain question as a control), scoring whether the model's top prediction gets every digit of the result. If 3- and 4-term accuracy still falls sharply while 2-term stays near 95%, this is a genuine capacity deficit with a clean assay. I can then trace it to components using that assay. If random chains are fine, the failures come from context, and the mechanism is different.

```python
import numpy as np, json, random
print([ (t) for t in tok.encode('Then $T = 12 + 7 = 19$.\n',add_special_tokens=False).tokens])
print([ (t) for t in tok.encode('total:** $12 + 7 = $19.\n\n',add_special_tokens=False).tokens])
rng=random.Random(20261105)
dev=json.load(open('/work/dev_ids.json'))
fmt=open('/practice/eval_prompt_format.txt').read()
def magn(kind):
    if kind=='d2': return rng.randint(11,99)
    if kind=='money': return rng.choice([5,10,12,15,20,24,25,30,35,36,40,45,48,50,60,64,75,80,90,120,150,200])
    if kind=='d3': return rng.randint(101,999)
def make(op,k,m):
    if op=='add':
        v=[magn(m) for _ in range(k)]; return ' + '.join(map(str,v)),sum(v),v
    if op=='sub':
        v=[magn(m) for _ in range(k)]; v[0]=sum(v[1:])+rng.randint(5,60)
        return ' - '.join(map(str,v)),v[0]-sum(v[1:]),v
    if op=='mul1':  # a * b + c
        a,b,c=rng.randint(2,12),rng.randint(2,40),magn(m); return f'{a} * {b} + {c}',a*b+c,[a,b,c]
    if op=='mul2':  # a * b + c * d
        a,b,c,d=rng.randint(2,12),rng.randint(2,40),rng.randint(2,12),rng.randint(2,40); return f'{a} * {b} + {c} * {d}',a*b+c*d,[a,b,c,d]
cells=[]
for k in [2,3,4,5]:
    for m in ['d2','money']: cells.append(('add',k,m,20))
cells+=[('add',2,'d3',20),('add',3,'d3',20),('add',4,'d3',20)]
for k in [2,3,4]: cells.append(('sub',k,'money',20))
cells+=[('mul1',3,'money',25),('mul2',4,'money',25)]
texts=[];idx=[]
for frame in ['H','L','Q']:
    for op,k,m,n in cells:
        for _ in range(n):
            e,val,v=make(op,k,m)
            if frame=='H':
                pre=fmt.replace('{problem}',prac[rng.choice(dev)]['problem'])+"Here's how to solve the problem step-by-step:\n\n1. **Calculate the total:** "+e+" ="
                post='.\n\n'
            elif frame=='L':
                pre=fmt.replace('{problem}',prac[rng.choice(dev)]['problem'])+"Let $T$ be the total.\nThen $T = "+e.replace('*','\\times')+" ="
                post='$.\n'
            else:
                pre='<bos><start_of_turn>user\nWhat is '+e+'?<end_of_turn>\n<start_of_turn>model\n'+e+' ='
                post='\n'
            ans=' '+str(val)
            full=pre+ans+post
            np_=len(tok.encode(pre,add_special_tokens=False).ids)
            texts.append(full); idx.append(dict(frame=frame,op=op,k=k,m=m,val=val,terms=v,n_pre=np_,ans=str(val)))
L=[len(tok.encode(t,add_special_tokens=False).ids) for t in texts]
print(len(texts),'max len',max(L),'total',sum(L))
print(repr(texts[0][-90:])); print(repr(texts[350][-70:])); print(repr(texts[-1][-60:]))
json.dump(texts,open('/work/tf_chain_texts.json','w')); json.dump(idx,open('/work/tf_chain_index.json','w'))
open('/work/notes.md','a').write("""
Run 65: chains acc by terms 3:.50, 4:.27, 5:0/7; of 104 wrong add/sub chains 23 exact one-term drop (null 1.7%), 55 contiguous partial sum.
Pass 14 (teacher-forced, ORIGINAL, no edit): /work/tf_chain_texts.json, index tf_chain_index.json. Fresh random chains in frames H, L, Q (plain question), add k2-5, d2/money/d3, sub k2-4, mixed. Random DEV problems as context. Purpose: is the chain deficit real on context-free chains? Score: top-1 on all result digits.
""")
print(edited())
```