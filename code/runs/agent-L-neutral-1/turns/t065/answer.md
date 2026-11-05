**What the last result showed.** The obvious explanation failed. The time deficit appears everywhere I split it: minutes only (-11), hours only (-16), both (-12), and at nearly the same size whether or not the reference uses 60 (-12.7 vs -13.9). So this isn't a failure to convert by 60. The week and per/each deficits also persist with time problems removed (-11.5 and -7.8), so time isn't a single driver of them all.

The problems I read fail in varied ways: a time gap ignored, wrong totals, and misreading what the question asks. One example stands out: the jet answer states `Distance = Speed * Time`, then writes `290 miles / 10 hours`, contradicting its own formula. That could be a unit-collocation slip, with "miles / hours" overriding the intended operator, but it is one example.

**What I conclude.** The deficit is real, but its cause is still open. There's a dull explanation that fits: rate problems may be conceptually harder than their step counts suggest, in which case there's no defect to repair. I shouldn't call this a mechanism until I can separate "the word itself disturbs something" from "the structure is hard".

**What I will do now.** First, two checks on existing data:
- Is the deficit visible at the token level on references, for number choices in time vs other problems?
- Does the model use different operators than the references in these groups, for example adding where references multiply?

If there's a gap, the next step is a real causal test. `generate` only accepts practice ids, so I can't alter problems there, but teacher-forced passes accept any text. That lets me swap "hours"/"minutes" for neutral words in problem and solution, then compare number choices on the same structure.

**What would confirm or refute it.** A token-level gap, or a one-way operator shift, gives me something concrete to test. If number choices on references are equally good in time problems, the failure isn't in those local decisions, and the swap test would need a different metric. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nTime deficit broken down: min only -11, hour only -16, both -11.7, ref uses 60 -12.7 vs no 60 -13.9 => NOT a 60-conversion failure. week -11.5 and per/each -7.8 persist without time problems. May be conceptual difficulty, not a defect; needs minimal-pair test (swap unit words in TF).\n")
# 1. token-level on references by group, pass 2 refs (j-512 = row in df)
d0=df.reset_index(drop=True); pl0=d0.id.map(lambda i:prob[i]['problem'].lower())
g_time=pl0.str.contains(r'\bminutes?\b|\bhours?\b').values; g_week=pl0.str.contains(r'\bweeks?\b|\bmonths?\b|\byears?\b').values; g_rate=pl0.str.contains(r'\bper\b|\beach\b|\bevery\b').values
Ev['r']=Ev.j-512; Ev['time']=g_time[Ev.r.values]; Ev['week']=g_week[Ev.r.values]; Ev['rate']=g_rate[Ev.r.values]
Ev['steps']=d0.id.map(lambda i:nsteps(prob[i])).values[Ev.r.values]
def tabf(c):
    x_=Ev.groupby([c,'cp']).agg(n=('ok','size'),ok=('ok','mean')).round(3).reset_index(); x_['group']=c; return x_.rename(columns={c:'val'})
print(pd.concat([tabf('time'),tabf('week'),tabf('rate')]).to_string(index=False))
# per-problem number accuracy, stratified by steps, so long solutions do not dominate
pp=Ev.groupby('r').agg(ok=('ok','mean'),n=('ok','size'),steps=('steps','first'),time=('time','first')).reset_index()
pp['sb']=pd.cut(pp.steps,[0,2,3,4,99],labels=['1-2','3','4','5+'])
print(pp.groupby(['sb','time'],observed=True).ok.agg(['size','mean']).round(3).unstack().to_string())
# all-token mean logprob on references
tlp2=np.load('/work/fwd_2/token_logprobs.npy',mmap_mode='r')
mlp=[]
for j in range(512,1024):
    L_=int(L2[j]); arr=np.array(ids2[j,:L_]); s0=int(np.where(arr==105)[0][1]+3)
    v_=np.array(tlp2[j,s0-1:L_-1],dtype=np.float32); mlp.append(float(np.nanmean(v_)))
pp2=pd.DataFrame({'lp':mlp,'time':g_time,'steps':d0.id.map(lambda i:nsteps(prob[i])).values}); pp2['sb']=pd.cut(pp2.steps,[0,2,3,4,99],labels=['1-2','3','4','5+'])
print('ref mean logprob'); print(pp2.groupby(['sb','time'],observed=True).lp.mean().round(3).unstack().to_string())
# 2. operator use, references vs own finished answers, per answer
opc=lambda s:{'+':len(re.findall(r'\d\s*\+\s*\$?\d',s)),'-':len(re.findall(r'\d\s*-\s*\$?\d',s)),'*':len(re.findall(r'\d\s*(?:\*|x|\\times)\s*\$?\d',s)),'/':len(re.findall(r'\d\s*/\s*\$?\d|\\frac',s))}
def share(ss):
    c_=collections.Counter()
    for s in ss: c_.update(opc(s))
    t_=sum(c_.values()) or 1
    return {k_:round(c_[k_]/t_,3) for k_ in '+-*/'}|{'n_ops':t_}
fin=~A2.cut
for nm_,m_ in [('time',tu),('week',wk&~tu),('rate only',pe&~tu&~wk),('rest',~tu&~wk&~pe)]:
    print(nm_,'n',int((m_&fin).sum()),'ref',share(A2.id[m_&fin].map(lambda i:prob[i]['solution'].split('####')[0])),'own',share(A2.text[m_&fin]))
```