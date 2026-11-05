**What the last result showed.** The recount weakened the over-adding story. `+` share in time problems is now 25.5% vs 17.7%, and in rate problems 23.6% vs 18.9%. So some excess remains, but it's much smaller than the first table showed, and week problems show none. As I suspected, part of that gap was a parsing artifact. Spurious `+` is still linked with low accuracy, but in hard groups nearly everything is, and overuse of "total" can't be read much, because references are terse and own answers are wordy everywhere.

The larger asymmetry now is division. In every group, "missing /" (16-21% of finished answers) far exceeds "spurious /" (2-4%), a ratio of about 5:1, and accuracy is lower when it's missing.

**A caution.** I've already been misled twice by this kind of table, and this one has a visible flaw. My `/` pattern requires spaces around the slash, so `250/100 = 2.50`, which appears in a sample above, doesn't count as division. Fractions written `1/2 * 20`, or "half of 20", wouldn't count either. Since the reference pattern is tight, the "missing /" figure could be largely my own miss. I won't draw a conclusion until that's fixed.

**What I will do now.** I'll recount using the union of the tight and unit-tolerant patterns for own answers. I'll also credit division expressed as "half", "third" or a fraction, so an equivalent way of dividing isn't scored as missing. The same missing and spurious rates are then reported per operator, pooled across groups, with a simple binomial comparison of missing vs spurious.

**What would confirm or refute it.** If `/` still shows a clear one-way shortfall after those corrections, I'll inspect the operator decision itself. That means asking whether `/` is near the top of the model's choices at the point where it took another operator. If the asymmetry falls to about balanced, then division is another artifact, and I'll stop building on operator counts. No forward pass is used.

```python
tighta={'+':re.compile(r'\d\s*\+\s*\$?\d'),'-':re.compile(r'\d\s*-\s*\$?\d'),'*':re.compile(r'\d\s*(?:\*|x|\\times|\\cdot|\u00d7)\s*\$?\d'),'/':re.compile(r'\d\s*/\s*\$?\d|\\frac|\\div|\u00f7')}
divwords=re.compile(r'\b(?:half|third|quarter|fourth|fifth|divide[sd]?|split|shared? equally)\b',re.I)
pctmul=re.compile(r'\b0?\.\d+\s*(?:\*|x|\\times|\\cdot)',re.I)
def ops_own2(s):
    c_=collections.Counter()
    for seg in re.split(r'\n|(?<=[a-z])\.\s',s):
        for k_ in '+-*/':
            if pats[k_].search(seg) or tighta[k_].search(seg): c_[k_]+=1
    return c_
def equiv_div(s): return bool(divwords.search(s)) or bool(pctmul.search(s))
oo2=F_.text.map(ops_own2); eq_=F_.text.map(equiv_div)
tb=[]
for nm_,m_ in G_.items():
    co=collections.Counter(); cr=collections.Counter()
    for x in oo2[m_]: co.update(x)
    for x in rf[m_]: cr.update(x)
    to=sum(co.values()) or 1; tr=sum(cr.values()) or 1
    d_={'group':nm_,'ops_own':to,'ops_ref':tr}
    for k_ in '+-*/': d_['own'+k_]=round(co[k_]/to,3); d_['ref'+k_]=round(cr[k_]/tr,3)
    tb.append(d_)
print(pd.DataFrame(tb).to_string(index=False))
from scipy.stats import binomtest
rec_=[]
for k_ in '+-*/':
    sp=np.array([(oo2[i_][k_]>0) and (rf[i_][k_]==0) for i_ in F_.index]); ms=np.array([(rf[i_][k_]>0) and (oo2[i_][k_]==0) for i_ in F_.index])
    if k_=='/': ms_c=ms&~eq_.values
    else: ms_c=ms
    c_=F_.correct.values
    a_,b_=int(ms_c.sum()),int(sp.sum())
    rec_.append({'op':k_,'missing':int(ms.sum()),'missing_credit':a_,'spurious':b_,'p_bal':round(binomtest(a_,a_+b_,0.5).pvalue,4),'acc_missing':round(float(c_[ms_c].mean()),3),'acc_spur':round(float(c_[sp].mean()),3),'acc_all':round(float(c_.mean()),3),'need_share':round(float(np.mean([rf[i_][k_]>0 for i_ in F_.index])),3)})
print(pd.DataFrame(rec_).to_string(index=False))
# hardness control: compare inside groups of the same reference step count
F_['ms_div']=np.array([(rf[i_]['/']>0) and (oo2[i_]['/']==0) for i_ in F_.index])&~eq_.values
F_['needs_div']=np.array([rf[i_]['/']>0 for i_ in F_.index])
F_['sb2']=pd.cut(F_.steps,[0,2,3,4,99],labels=['1-2','3','4','5+'])
print(F_[F_.needs_div].groupby(['sb2','ms_div'],observed=True).correct.agg(['size','mean']).round(3).unstack().to_string())
print('finished answers that need division: acc',round(float(F_[F_.needs_div].correct.mean()),3),'n',int(F_.needs_div.sum()),'| no division needed: acc',round(float(F_[~F_.needs_div].correct.mean()),3),'n',int((~F_.needs_div).sum()))
s_=F_[F_.ms_div&(F_.steps<=3)].sample(6,random_state=2)
for _,r_ in s_.iterrows():
    print('--- gold',r_.gold,'pred',r_.pred,'\n P:',r_.problem[:150].replace('\n',' '),'\n R:',prob[r_.id]['solution'][:120].replace('\n',' '),'\n A:',r_.text[-130:].replace('\n',' '))
```