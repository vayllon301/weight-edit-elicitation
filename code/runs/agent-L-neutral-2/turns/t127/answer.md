The data supports this: 249 problems have both a right and a wrong answer, which is promising. Next I'll build a prefix tree per problem and extract **decisive forks**, where one branch leads only to right answers and the other only to wrong ones. I'll do this twice: with all arms, and with only the arms that don't contain P1. The second set can test P1 fairly, because those forks weren't created by P1's own wins and losses. First-token forks are counted apart, since they're opener choices rather than reasoning. Forks at the stop token are also counted apart.

I'm also checking that each fork re-tokenizes exactly and matches how the earlier texts were built, so the token positions I read back will be right. No pass is spent yet. The count decides whether this screen has enough power.

```python
import numpy as np, collections, json
EOT=106
def build_forks(allowed):
    forks=[]
    for pid,v in recs.items():
        ans=[(a,ok,enc(t)) for a,ok,cu,t,nt in v if a in allowed]
        if len(set(x[1] for x in ans))<2: continue
        def walk(g,t):
            while True:
                ch=collections.defaultdict(list)
                for i in g:
                    tk=ans[i][2]; ch[tk[t] if t<len(tk) else EOT].append(i)
                if len(ch)>1: break
                g=list(ch.values())[0]; t+=1
                if t>len(ans[g[0]][2]): return
            for ka,ga in ch.items():
                if all(ans[i][1] for i in ga):
                    for kb,gb in ch.items():
                        if not any(ans[i][1] for i in gb):
                            forks.append(dict(id=pid,t=t,prefix=ans[ga[0]][2][:t],a=int(ka),b=int(kb),arms_a=sorted(set(ans[i][0] for i in ga)),arms_b=sorted(set(ans[i][0] for i in gb)),n_a=len(ga),n_b=len(gb)))
            for k,gg in ch.items():
                if k!=EOT and len(set(ans[i][1] for i in gg))==2: walk(gg,t+1)
        walk(list(range(len(ans))),0)
    return forks
ALLARM=set(a for v in recs.values() for a,*_ in v)
F_all=build_forks(ALLARM-{'OF'})
F_noP=build_forks(ALLARM-{'OF','P1','P1L2'})
def desc(F,name):
    t=np.array([f['t'] for f in F]); pr=len(set(f['id'] for f in F))
    print(name,'forks',len(F),'problems',pr,'t=0',int((t==0).sum()),'t<10',int((t<10).sum()),'median t',np.median(t),'stop forks',sum(1 for f in F if EOT in (f['a'],f['b'])))
desc(F_all,'all(noOF)'); desc(F_noP,'noP1')
print('dev_b share all',np.mean([f['id'] in set(db) for f in F_all]))
T=json.load(open('/work/tf_ref_texts.json')); ids=np.load('/work/fwd_3/input_ids.npy',mmap_mode='r')
print(repr(T[0][:70]),'first ids',ids[0,:6].tolist(), 'enc',enc(T[0])[:5])
f=F_all[len(F_all)//2]; print({k:(v if k!='prefix' else len(v)) for k,v in f.items()})
print('tok decode', repr(tok.decode([f['a']])[:20]), repr(tok.decode([f['b']])[:20]))
```