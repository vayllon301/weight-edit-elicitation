**What the last result showed.** This time the asymmetry held up after the correction. Even after crediting words like "half" and fractions, 58 answers miss a needed division against 18 with a spurious one, a clearly one-way result. Finished answers needing division score 18.4%, vs 31.2% for those that don't, and about half of finished answers need one. Over-adding also survives, more weakly (69 spurious vs 39 missing, p = 0.005). `-` and `*` are balanced.

The samples add a useful pattern. Kyle (60 is 3 times last year), Marta (work out hours from money), Tracy (`9 cups * 2.25 cups/pound`) and the earlier Randy case (`7 * 29 + 1`) all apply the relation forward when it must be inverted. So the candidate failure is **forward application: the model multiplies or adds the given numbers when the problem requires undoing an operation**.

**Caveats.** "Missing division gives about 0% accuracy" is partly circular, since skipping a required step is itself the error. And some problems may be solvable without an explicit division, so this isn't proven as a cause.

**What I will do now.** Before designing anything, I need to know where this decision is made and how close it is. Some cases are a direct operator error on the right numbers (Tracy), while others take a different path entirely (Mushrooms). I'll count equations where swapping only the operator would give the gold or a reference intermediate, broken down by used-vs-needed operator. Then, using the pass 2 predictions at those operator tokens, I'll measure how far the needed operator sits behind the chosen one.

**What would confirm or refute it.** Many fixable cases, one-way toward needing `/`, with small gaps, would give a sharp edit target at the operator token. Few fixable cases would mean that the decision is made earlier, at the plan level, and an operator-token edit couldn't reach it. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nOperator bias after corrected parsing and crediting 'half' etc: missing / 58 vs spurious 18 (one-way, p<.0001); + spurious 69 vs missing 39 (p=.005); -,* balanced. Finished needing / acc .184 vs .312 without; half of problems need /. Candidate failure: FORWARD APPLICATION (multiply/add givens when must invert). Partly circular; not yet causal.\n")
numu=r'\$?(\d[\d,]*(?:\.\d+)?)'
eq2=re.compile(numu+r"(?P<u1>[^\n\d=+*/]{0,24}?)\s*(?P<op>\+|-|\*|/|x|\\times|\\cdot|\u00d7)\s*"+r"(?P<u2>[^\n\d=]{0,3})"+numu+r"[^\n\d=]{0,24}?\s*=\s*"+r"\$?(\d[\d,]*(?:\.\d+)?)")
canon={'x':'*','\\times':'*','\\cdot':'*','\u00d7':'*'}
def apply(op,a,b):
    if op=='+': return a+b
    if op=='-': return a-b
    if op=='*': return a*b
    if op=='/': return a/b if b!=0 else None
def close(x,S): return x is not None and any(abs(x-v_)<=max(1e-6,abs(v_)*1e-3) for v_ in S)
rec_=[]
for i_,r_ in allr.iterrows():
    o=prob[r_.id]; sb_=o['solution'].split('####')[0]; g_=norm(str(r_.gold))
    tgt=set(v_ for v_ in (norm(x) for x in re.findall(r'=\s*\$?(-?\d[\d,]*\.?\d*)',sb_)) if v_ is not None)|({g_} if g_ is not None else set())
    if r_.correct>0: continue
    seen=set()
    for m_ in eq2.finditer(r_.text):
        pre=r_.text[:m_.start()].rstrip()
        if pre and pre[-1] in '+-*/)' : continue
        a=float(m_.group(1).replace(',','')); b=float(m_.group(5).replace(',','')); c=float(m_.group(6).replace(',',''))
        op=canon.get(m_.group('op'),m_.group('op'))
        if (a,op,b) in seen: continue
        seen.add((a,op,b))
        if close(apply(op,a,b),tgt): continue
        alts=[(k_,apply(k_,a,b)) for k_ in '+-*/' if k_!=op]
        alts+= [('/r',apply('/',b,a)),('-r',apply('-',b,a))]
        hit=[k_ for k_,v_ in alts if close(v_,tgt) and v_ not in (a,b)]
        if hit:
            rec_.append({'i':i_,'used':op,'need':hit[0][0],'rev':'r' in hit[0],'opchar':m_.start('op'),'a':a,'b':b,'mode':mode_of(r_.text),'gold_alt':close(dict(alts).get(hit[0]),{g_}),'cut':bool(r_.cut)})
Fx=pd.DataFrame(rec_); print('flip-fixable equations',len(Fx),'answers',Fx.i.nunique(),'of wrong',int((allr.correct==0).sum()),'| direct to gold, answers',Fx[Fx.gold_alt].i.nunique())
print(pd.crosstab(Fx.used,Fx.need).to_string())
first=Fx.sort_values('opchar').groupby('i').head(1); print('first per answer'); print(pd.crosstab(first.used,first.need).to_string())
# gaps at the operator token in own answers, pass 2 (first 512 answers only)
tmp=Fx[Fx.i<512]
lab2ids=collections.defaultdict(list)
for ch_ in ['+','-','*','/','x','\u00d7','\\times','\\cdot']:
    for pre_ in ['',' ']:
        e_=tok.encode(pre_+ch_,add_special_tokens=False).ids
        if len(e_)==1: lab2ids[canon.get(ch_,ch_)].append(e_[0])
print('op ids',{k_:len(v_) for k_,v_ in lab2ids.items()})
gapr=[]
tk2=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); lpk2=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r')
for _,r_ in tmp.iterrows():
    i_=int(r_.i); txt=tf2[i_]; off=len(txt)-len(df.text.iloc[i_])+int(r_.opchar)
    enc=tok.encode(txt,add_special_tokens=False); tix=next((q_ for q_,(a_,b_) in enumerate(enc.offsets) if a_<=off<b_),None)
    if tix is None: continue
    tix2=tix
    ids_=np.array(tk2[i_,tix2-1]); lps=np.array(lpk2[i_,tix2-1],dtype=np.float32)
    need=r_.need[0]; use=r_.used
    pu=[lps[np.where(ids_==x)[0][0]] for x in lab2ids[use] if x in ids_]; pn=[lps[np.where(ids_==x)[0][0]] for x in lab2ids.get(need,[]) if x in ids_]
    gapr.append({'used':use,'need':need,'pu':max(pu) if pu else np.nan,'pn':max(pn) if pn else np.nan,'in_top10':bool(pn),'used_top1':int(ids_[0]) in lab2ids[use]})
Gp=pd.DataFrame(gapr); Gp['gap']=Gp.pu-Gp.pn
print('n',len(Gp),'used op is top1',round(float(Gp.used_top1.mean()),3),'need op in top10',round(float(Gp.in_top10.mean()),3))
print(Gp.groupby(['used','need']).agg(n=('gap','size'),in10=('in_top10','mean'),med_gap=('gap','median')).round(2).to_string())
print('gap pct where measured',np.round(np.nanpercentile(Gp.gap,[10,25,50,75,90]),2),'share gap<1',round(float((Gp.gap<1).mean()),3),'share gap<2',round(float((Gp.gap<2).mean()),3))
```