`<bos>` removal scored 59.77 with 16.21% cut, against 61.91 and 16.60 for Edit A alone. That's a raw drop of 2.14 points with hardly any change in cut-offs, which doesn't support the idea that the sink crowds out useful reading. If anything, the sink looks like a mildly useful constant.

Next I'll compute the paired difference and restore `/work/edited` to Edit A so a dead edit can't carry forward. If the paired confidence interval excludes a meaningful gain, I'll close this line without testing other strengths.

I'll also start the fork analysis, which needs no pass. dev_b now has around ten answers per problem on the `Let` path, so I can build a prefix tree for each problem and look for the forks where one branch mostly leads to right answers and the other to wrong ones. Those are the real decision points, unlike the first divergence, which is often just rephrasing. If decisive forks cluster into one decision type, such as which quantity gets chosen or how a phrase is read, I'd have a specific failure to target. If they're scattered, that strengthens the view that no single mechanism is at work.

```python
import json, numpy as np, re, collections, shutil, os
def rd(p):
    return {json.loads(l)['id']:json.loads(l) for l in open(p)}
g6=rd('/work/gen_6.jsonl'); g23=rd('/work/gen_23.jsonl')
ids=sorted(set(g6)&set(g23))
c6=np.array([g6[i]['correct'] for i in ids],float); c23=np.array([g23[i]['correct'] for i in ids],float)
d=c23-c6
print('pass23 vs a=1: n',len(ids),'d %.2f se %.2f'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d))),'W>R',int((d>0).sum()),'R>W',int((d<0).sum()),'changed',sum(g6[i]['text']!=g23[i]['text'] for i in ids))
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct23=np.array([g23[i]['cut'] for i in ids],bool)
both=~ct6&~ct23
print('both finished n',int(both.sum()),'d %.2f se %.2f'%(100*d[both].mean(),100*d[both].std(ddof=1)/np.sqrt(both.sum())),'| cut->fin R',int(((ct6&~ct23)&(c23==1)).sum()),'of',int((ct6&~ct23).sum()),'| fin->cut lost R',int(((~ct6&ct23)&(c6==1)).sum()),'of',int((~ct6&ct23).sum()))
# restore Edit A
drop_edit('model.layers.23.self_attn.v_proj.weight')
print('edited now',edited(), 'L17==best_1', np.array_equal(np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy'),np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy')))
with open('/work/notes.md','a') as f:
    f.write("\nPass 23 result: <bos> value removal L23 dev_b 59.77 cut 16.21 tokens 356.3 (a=1: 61.91, 16.60); paired d %.2f se %.2f. Sink is not a crowding hindrance. /work/edited RESTORED to Edit A a=1.\n"%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d))))
# Fork analysis over Let-path arms on dev_b
arms=['gen_5','gen_6','gen_7','gen_8','gen_10','gen_11','gen_15','gen_16','gen_17','gen_23']
A={}
for a in arms:
    p='/work/%s.jsonl'%a
    if os.path.exists(p): A[a]=rd(p)
print('arms',list(A), {a:len(v) for a,v in A.items()})
DB=json.load(open('/work/dev_b.json'))
def toks(t): return tok.encode(t,add_special_tokens=False).ids if hasattr(tok.encode(t,add_special_tokens=False),'ids') else tok.encode(t,add_special_tokens=False)
forks=[]; conc=collections.Counter()
for pid in DB:
    ans={}
    for a,v in A.items():
        if pid in v:
            r=v[pid]; ans.setdefault(r['text'],int(r['correct']))
    conc[(len(ans)>1, )]+=0
    items=list(ans.items())
    if len(items)<4: continue
    T=[(toks(t),c) for t,c in items]
    def rec(group,depth):
        if len(group)<2: return
        cs=[c for _,c in group]
        if min(cs)==max(cs): return
        d=depth
        while True:
            nxt=collections.defaultdict(list)
            for s,c in group:
                nxt[s[d] if d<len(s) else -1].append((s,c))
            if len(nxt)>1: break
            d+=1
        branches=sorted(nxt.items(),key=lambda kv:-len(kv[1]))
        rates=[(k,len(b),np.mean([c for _,c in b])) for k,b in branches]
        forks.append(dict(pid=pid,depth=d,prefix=group[0][0][:d],rates=rates,n=len(group),rate=np.mean(cs)))
        for k,b in branches: rec(b,d+1)
    rec(T,0)
print('problems with >=4 distinct answers',len({f['pid'] for f in forks}),'forks',len(forks))
# decisive: at least 2 answers on each of two branches, one branch all right, other all wrong
dec=[]
for f in forks:
    R=[(k,n) for k,n,r in f['rates'] if r==1 and n>=2 and k!=-1]; W=[(k,n) for k,n,r in f['rates'] if r==0 and n>=2 and k!=-1]
    if R and W: dec.append((f,R[0],W[0]))
print('decisive forks',len(dec),'in problems',len({f['pid'] for f,_,_ in dec}))
json.dump([dict(pid=f['pid'],depth=f['depth'],prefix=f['prefix'],R=R,W=W,n=f['n']) for f,R,W in dec],open('/work/decisive_forks.json','w'))
def cat(t):
    s=tok.decode([t])
    if re.fullmatch(r'\s*\d+',s): return 'digit'
    if re.fullmatch(r'\s*[\+\-\*/=\\\(\)\^]+',s) or s.strip() in ('\\times','\\frac','\\cdot'): return 'op'
    if s.strip()=='' : return 'space'
    if re.fullmatch(r'\s*[A-Za-z\']+',s): return 'word'
    return 'other'
cr=collections.Counter(); cw=collections.Counter(); pair=collections.Counter()
for f,R,W in dec:
    a,b=cat(R[0]),cat(W[0]); cr[a]+=1; cw[b]+=1; pair[(a,b)]+=1
print('R cat',cr.most_common(),'\nW cat',cw.most_common(),'\npairs',pair.most_common(8))
print('depth pct 10/50/90',np.percentile([f['depth'] for f,_,_ in dec],[10,50,90]) if dec else None)
```