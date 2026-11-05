**What the last result showed.** x1038 barely differs between groups. At the same layer, right, wrong and cut answers stay within about 5% of each other, and its answer-averaged CV is only 2-5%. At the last prompt token the groups match even more closely. So x1038 doesn't explain failure at the per-answer level, and I'm dropping it as a failure mechanism. Being near-constant, it could still be useful as a source for writing a fixed vector.

**Mode findings.** The mode gap is more interesting. "Let" answers reach 32.4% accuracy when they finish (n=185) against 21.3% for "Here's" (n=178). However, "Let" also gets cut more often (33% vs 23%), so overall accuracy is 21.7% vs 17.0%, a gap of about 1.3 standard errors. Answers that reach "Final Answer" score 31%, but that split is selected on finishing, so it proves little. Nor does the marker itself help: within "Here's", accuracy is 18% with it and 15% without.

**What I will do now.** I need to know whether the mode gap survives a control for difficulty, and what actually happens where answers loop. I'll stratify mode accuracy by reference step count, which is a difficulty measure neither mode can influence. I'll also read how the loops begin in each mode, since the 145 cut answers are the largest block of near-certain failures.

**What would confirm or refute it.** If "Let" keeps its lead within each step count, then a causal mode-flip test is worth a pass. If the gap disappears, the mode line closes. For loops, a common entry pattern, such as re-reading or re-deriving at a contradiction, would point to a specific point of failure. Scattered entries would suggest loops are only a symptom. No forward pass is used.

```python
import pickle
with open('/work/notes.md','a') as f:
    f.write("\nx1038 (huge near-const residual coord, 80-98% of energy L8+; preff weight -1): right/wrong/cut within ~5% at same layer, answer-mean CV 2-5% => NOT a failure mechanism, usable as near-constant source only.\nBaseline gen_1: acc .193, cut 28%, wrong finished 269/512. Modes: heres n=230 acc .170 cut .226; let n=277 acc .217 cut .332; finished acc heres .213 let .324.\n")
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    o=json.loads(l); prob[o['id']]=o
print('keys',list(next(iter(prob.values())).keys()))
sol_key=[k for k in next(iter(prob.values())).keys() if k not in('id','source','problem','answer')][0]
def nsteps(o):
    s=o[sol_key]
    return max(1,len([x for x in s.split('\n') if x.strip() and not x.strip().startswith('####')]))
df['steps']=df.id.map(lambda i:nsteps(prob[i]))
df['plen']=df.id.map(lambda i:len(prob[i]['problem'].split()))
print('steps dist',df.steps.value_counts().sort_index().to_dict())
df['sb']=pd.cut(df.steps,[0,2,3,4,99],labels=['1-2','3','4','5+'])
t_=df[df['mode']!='other'].groupby(['sb','mode'],observed=True).agg(n=('correct','size'),acc=('correct','mean'),cut=('cut','mean'))
print(t_.round(3).unstack().to_string())
# Mantel-Haenszel style stratified difference (let-heres)
num_=0;den_=0;var_=0
for sb,g_ in df[df['mode']!='other'].groupby('sb',observed=True):
    a_=g_[g_['mode']=='let'].correct; b_=g_[g_['mode']=='heres'].correct
    w_=len(a_)*len(b_)/(len(a_)+len(b_)); d_=a_.mean()-b_.mean()
    num_+=w_*d_; den_+=w_; var_+=w_**2*(a_.var(ddof=1)/len(a_)+b_.var(ddof=1)/len(b_))
print('stratified let-heres diff',round(100*num_/den_,1),'+-',round(100*np.sqrt(var_)/den_,1),'pt')
print('mode mix by steps',df.groupby('sb',observed=True)['mode'].apply(lambda s:round((s=='let').mean(),3)).to_dict())
# loop entry reading
def find_loop(s):
    # smallest period p (chars) with the tail repeating at least 3 times
    n=len(s); tail=s[-3000:] if n>3000 else s
    for p in range(20,900):
        if tail[-p:]==tail[-2*p:-p]==tail[-3*p:-2*p]:
            return p
    return None
lr=[]
for idx,r_ in df[df.cut].iterrows():
    s=r_.text; p=find_loop(s)
    if p is None: lr.append((idx,None,None)); continue
    i=len(s)-p
    while i-p>=0 and s[i-p:i]==s[i:i+p]: i-=p
    # walk back char by char to the true start of the periodic part
    while i>0 and s[i-1]==s[i-1+p]: i-=1
    lr.append((idx,p,i))
lrd=pd.DataFrame(lr,columns=['idx','period','start']).set_index('idx')
print('cut',len(lrd),'periodic',int(lrd.period.notna().sum()),'median period chars',lrd.period.median(),'median start frac',round(float((lrd.start/df.loc[lrd.index].text.str.len()).median()),3))
print('by mode',df.loc[lrd.index].groupby('mode').size().to_dict())
ent=collections.Counter()
for idx,r_ in lrd.dropna().iterrows():
    s=df.loc[idx].text; st=int(r_.start)
    ent[' '.join(s[st:st+40].split()[:2])]+=1
print('loop first words',ent.most_common(12))
samp=lrd.dropna().sample(6,random_state=3)
for idx,r_ in samp.iterrows():
    s=df.loc[idx].text; st=int(r_.start); print(df.loc[idx,'mode'],int(r_.period),'PRE',repr(s[max(0,st-70):st]),'CYCLE',repr(s[st:st+90]))
```