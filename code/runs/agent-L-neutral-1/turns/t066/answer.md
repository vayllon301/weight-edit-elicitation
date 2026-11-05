**What the last result showed.** This narrows down where the deficit lives.

On reference solutions, number choices are just as good in time problems as elsewhere: copied numbers are 0.838 vs 0.829, and per-problem accuracy matches within each step stratum. New numbers are slightly lower (0.785 vs 0.824), but that isn't enough to explain a 14-point loss. So given the correct path, the model follows it about as well in time problems. The failure must therefore lie earlier, in deciding what to compute, not in carrying out each step.

The operator table fits that idea. In multiplicative groups the model adds more and multiplies less than the references: time `+` is 34% vs 21% and `*` 42% vs 49%, and rate problems show `+` 31% vs 22%. In the remaining problems `+` matches the references at 36-37%. Several examples I read show the same pattern of summing parts into a "total".

**A measurement problem.** I can't trust this table yet. My own-answer regex needs a digit right next to the operator, so it misses `3 hours/day * 7 days`, which is why own answers show only 167 operators against 480 for references. It may also inflate `+`, because a chain counts several times.

**What I will do now.** I'll recount operators allowing units between the numbers and counting each chain once per equation. Then I'll use a fairer per-problem measure: "spurious" operators the model used but the reference never needed, and "missing" ones the reference needed but the model never used, by group and for right vs wrong answers. I'll also check how often "total" appears.

**What would confirm or refute it.** The over-adding idea survives only if spurious `+` remains clearly more common than other spurious operators in rate/time problems after the recount, and answers with it are more often wrong. If the gap shrinks to nothing, it was a parsing artifact and I'll say so. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nTime deficit is NOT in local number choice on refs (copy .838 vs .829; per-problem equal within steps) => failure is earlier, in deciding what to compute. Op share own vs ref suggests over-adding in rate/time groups but regex misses units; recounting.\n")
unit=r"[^\n\d=+*/]{0,24}?"
pats={'+':re.compile(r'\d'+unit+r'\s\+\s[^\n\d=]{0,3}\d'),'-':re.compile(r'\d'+unit+r'\s-\s[^\n\d=]{0,3}\d'),'*':re.compile(r'\d'+unit+r'\s(?:\*|x|\\times|\\cdot|\u00d7)\s[^\n\d=]{0,3}\d'),'/':re.compile(r'\d'+unit+r'\s/\s[^\n\d=]{0,3}\d|\\frac')}
tight={'+':re.compile(r'\d\s*\+\s*\$?\d'),'-':re.compile(r'\d\s*-\s*\$?\d'),'*':re.compile(r'\d\s*[*x]\s*\$?\d'),'/':re.compile(r'\d\s*/\s*\$?\d')}
def ops_own(s):
    # one count per line-level equation segment for each operator: chains count once
    c_=collections.Counter()
    for seg in re.split(r'\n|(?<=[a-z])\.\s',s):
        for k_,p_ in pats.items():
            if p_.search(seg): c_[k_]+=1
    return c_
def ops_ref(s):
    c_=collections.Counter()
    for seg in s.split('\n'):
        for k_,p_ in tight.items():
            if p_.search(seg): c_[k_]+=1
    return c_
F_=A2[~A2.cut].copy()
oo=F_.text.map(ops_own); rf=F_.id.map(lambda i:ops_ref(prob[i]['solution'].split('####')[0]))
pl2=F_.problem.str.lower(); tu2=pl2.str.contains(r'\bminutes?\b|\bhours?\b'); wk2=pl2.str.contains(r'\bweeks?\b|\bmonths?\b|\byears?\b'); pe2=pl2.str.contains(r'\bper\b|\beach\b|\bevery\b')
G_={'time':tu2,'week':wk2&~tu2,'rate only':pe2&~tu2&~wk2,'rest':~tu2&~wk2&~pe2}
tb=[]
for nm_,m_ in G_.items():
    co=collections.Counter(); cr=collections.Counter()
    for x in oo[m_]: co.update(x)
    for x in rf[m_]: cr.update(x)
    to=sum(co.values()) or 1; tr=sum(cr.values()) or 1
    d_={'group':nm_,'n':int(m_.sum()),'ops_own':to,'ops_ref':tr}
    for k_ in '+-*/': d_['own'+k_]=round(co[k_]/to,3); d_['ref'+k_]=round(cr[k_]/tr,3)
    tb.append(d_)
print(pd.DataFrame(tb).to_string(index=False))
rec_=[]
for nm_,m_ in G_.items():
    for k_ in '+-*/':
        sp=[(oo[i_][k_]>0) and (rf[i_][k_]==0) for i_ in F_.index[m_]]; ms=[(rf[i_][k_]>0) and (oo[i_][k_]==0) for i_ in F_.index[m_]]
        c_=F_.correct[m_].values; sp=np.array(sp); ms=np.array(ms)
        rec_.append({'group':nm_,'op':k_,'spurious':round(sp.mean(),3),'acc_sp':round(float(c_[sp].mean()),3) if sp.any() else None,'acc_nosp':round(float(c_[~sp].mean()),3) if (~sp).any() else None,'missing':round(ms.mean(),3),'acc_ms':round(float(c_[ms].mean()),3) if ms.any() else None,'acc_nomiss':round(float(c_[~ms].mean()),3) if (~ms).any() else None})
print(pd.DataFrame(rec_).to_string(index=False))
tot_=F_.text.str.lower().str.count(r'\btotal\b'); F_['tot']=tot_; F_['reftot']=F_.id.map(lambda i:len(re.findall(r'\btotal\b',prob[i]['solution'].lower()))); F_['q_total']=F_.problem.str.lower().str.contains(r'\btotal\b|altogether|in all|combined')
print('own "total" per answer by group',{k_:round(float(tot_[m_].mean()),2) for k_,m_ in G_.items()},'ref',{k_:round(float(F_.reftot[m_].mean()),2) for k_,m_ in G_.items()})
print('problem asks total',F_.groupby('q_total').agg(n=('correct','size'),acc=('correct','mean'),own=('tot','mean')).round(3).to_dict('index'))
```