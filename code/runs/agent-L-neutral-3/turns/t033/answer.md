The causal test gave a negative result for E2. Boosting the retrieval head by 1.5 left `fromproblem_first` accuracy unchanged (-0.13 ± 0.17 points) and slightly hurt result digits (-0.5 points). So this head isn't limited by its output share. When retrieval fails, it probably points to the wrong place rather than being too weak, and scaling can't fix that. The correlation I saw was real but not a lever. I'm dropping E2. Next I want to widen the search with two free text analyses. First, `othernum` is the second-biggest break class, so I'll check whether it's mainly number words ("three", "twice", "half", "dozen") that a digit-oriented retrieval circuit could miss, and how much accuracy differs. Second, Here versus Let didn't matter, but the reference solutions use a different plain-sentence style that often opens with "First". I'll look at which alternative openers the model gives non-trivial probability, and at the share of reference openers, to judge whether a third style could be worth testing. A clear weakness on number words would be a specific new circuit target, while a viable third opener would be worth one generation pass.

```python
drop_edit('model.layers.17.self_attn.o_proj.weight'); print('edited',edited())
open('/work/notes.md','a').write("""
Pass 6 E2 (L17 h0 x1.5, TF refs): fromproblem_first -0.13+-0.17 pts, result_first -0.49+-0.20. BOOST DOES NOT HELP; head correlates with success (peak 47.7% vs 14.7%) but not limited by gain. E2 dropped.
""")
# 1. Word numbers
W={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'fifteen':15,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'hundred':100,'thousand':1000,'twice':2,'double':2,'half':2,'triple':3,'thrice':3,'quarter':4,'third':3,'dozen':12}
imp={'day':[24],'hour':[60],'minute':[60],'week':[7],'year':[12,52,365],'month':[30,4],'foot':[12],'feet':[12],'percent':[100],'%':[100]}
A_=load_fwd('/work/fwd_4/')
sub=collections.defaultdict(lambda:[0,0])
i_cur=None
for (i,x,c) in CL:
    if c!='othernum_first': continue
    t=tm[i]; ids_=A_['ii'][i]
    r=x
    while r<t['tlen'] and int(ids_[r]) in DIG: r+=1
    s=''.join(DIG[int(ids_[y])] for y in range(x,r))
    pb=byid[t['id']]['problem'].lower(); wds=set(re.findall(r"[a-z]+",pb))
    v=int(s) if s.isdigit() else -1
    isw=any(W[w]==v for w in wds if w in W)
    isimp=any(v in imp[k] for k in imp if k in pb)
    k_='wordnum' if isw else ('unit' if isimp else 'derived')
    tops=[int(a) for a in A_['ti'][i,x-1]]; dg=[a for a in tops if a in DIG]
    good=bool(dg) and dg[0]==int(ids_[x])
    sub[k_][0]+=1; sub[k_][1]+=good
print({k:(v[0],round(v[1]/v[0],3)) for k,v in sub.items()})
# 2. Openers in references and alternatives in the model's first-token distribution
ref_open=collections.Counter(clean_ref(byid[p]['solution']).split()[0] for p in ids800)
print('ref first word top',[(norm(k),v) for k,v in ref_open.most_common(12)])
ff=collections.Counter()
for i,t in enumerate(tm):
    if t['kind']!='ref': continue
    p=t['plen']
    for a,b in zip(A_['ti'][i,p-1],A_['tp'][i,p-1]):
        ff[int(a)]+=float(np.exp(b))/800
print('model first-token mean prob',[(norm(Dc([k])).strip(),round(v,5)) for k,v in ff.most_common(8)])
print(' mean logprob of First',[round(float(np.mean([v for v in [float(A_['tp'][i,tm[i]['plen']-1][list(A_['ti'][i,tm[i]['plen']-1]).index(E('First')[0])] ) for i in range(800) if E('First')[0] in list(A_['ti'][i,tm[i]['plen']-1])] ])),2) if any(E('First')[0] in list(A_['ti'][i,tm[i]['plen']-1]) for i in range(800)) else None])
```