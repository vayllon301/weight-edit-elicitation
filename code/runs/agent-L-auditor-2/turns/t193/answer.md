The remaining failures now look like ordinary reasoning mistakes. On the "Let" and "Here" openers, keep-top is about equal (38.4% and 37.0%), and most of the gain happens with the opener unchanged (21.6% to 37.0% on 408 problems). So mode isn't the lever, and that line stays closed. The long failures simply miss part of the problem. One never converts ounces, and another answers "1" after tracking the wrong quantity.

Two of those excerpts also ended without the `####` marker. The grader then takes the last number in the text, so that could cost some correct answers. My notes said format wasn't the gap in the original, but I should recheck that at this new operating point.

Next I'll look for evidence of a second component rather than guess, using three checks that cost no budget:

- **Co-writers:** does any other head write into the same stream subspace as h3's harmful tail? If so, readers downstream might still be getting that signal, and that head becomes a concrete candidate.
- **Marker loss:** how many answers are lost to missing `####`?
- **Earlier levers:** "solve pull +" lost accuracy because it added loops, and loops are now rare, so that trade-off may have changed. I'll reread what that edit was.

A head with clearly more overlap than the rest would earn a test. If none stands out, h3's harm is probably not spread across heads.

```python
import json, re, numpy as np, torch
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
K={**rd('/work/gen_35_edited.jsonl'),**rd('/work/gen_34_edited.jsonl')}
O={**rd('/work/gen_35_original.jsonl'),**rd('/work/gen_34_original.jsonl')}
ids=sorted(json.load(open('/work/dev_fe.json')))
for lab,M in (('orig',O),('keep1',K)):
    um=[i for i in ids if M[i]['used_marker']]; nm=[i for i in ids if not M[i]['used_marker'] and not M[i]['cut']]
    lost=0
    for i in nm:
        if M[i]['correct']: continue
        tail_=M[i]['text'].replace(',','')[-120:]
        lost+= str(M[i]['gold']).replace('.0','') in re.findall(r'\d+(?:\.\d+)?',tail_)
    print(lab,'marker n',len(um),'acc',round(np.mean([M[i]['correct'] for i in um]),3),'| no marker finished n',len(nm),'acc',round(np.mean([M[i]['correct'] for i in nm]) if nm else 0,3),'wrong with gold in last 120 chars',lost)

# co-writers of h3 tail, in gain-weighted stream space, register excluded
def blk(L):
    O_=load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    g=(1+load_original(f'model.layers.{L}.post_attention_layernorm.weight').float()).abs()
    g=g.clone(); g[1038]=0
    return O_,g
O11,g11=blk(11)
B=O11[:,768:1024]; U,S,Vh=torch.linalg.svd(B,full_matrices=False)
tail=B-S[0]*torch.outer(U[:,0],Vh[0])
Ut,St,_=torch.linalg.svd(tail*g11[:,None],full_matrices=False)
print('gain-weighted tail top sv',[round(float(x),1) for x in St[:6]],'energy top8/32',round(float((St[:8]**2).sum()/(St**2).sum()),3),round(float((St[:32]**2).sum()/(St**2).sum()),3))
res=[]
for L in range(26):
    O_,g=blk(L)
    for h in range(4):
        X=O_[:,256*h:256*(h+1)]*g[:,None]
        e=float((X**2).sum())
        r=[float(((Ut[:,:k].T@X)**2).sum())/e/(k/1151) for k in (8,32)]
        res.append((L,h,r[0],r[1]))
res=np.array(res)
print('enrichment over chance: median k8 k32',round(float(np.median(res[:,2])),2),round(float(np.median(res[:,3])),2))
print('top by k32',[(int(a),int(b),round(c,1),round(d,1)) for a,b,c,d in res[np.argsort(-res[:,3])[:12]]])
print('L11 heads',[(int(b),round(c,1),round(d,1)) for a,b,c,d in res[res[:,0]==11]])

txt=open('/work/notes.md').read().split('\n')
sel=[l[:300] for l in txt if re.search(r'solve|Solve|SOLVE',l)]
print('\n'.join(sel[:12])[:3000])
```