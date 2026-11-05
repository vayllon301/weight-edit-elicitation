Replication is finished, and the answer is clear. c3 gives -0.78 ± 1.56 (7 vs 9). The four estimates (+1.95, +1.17, +0.39, -0.78) pool to **33 wrong-to-right against 26 right-to-wrong over 1,024 answers, or +0.68 ± 0.75 points**. Cut is unchanged, 270 answers against 272. So `cand_x33_b0` shows no demonstrated gain, and the promising first pass was mostly selection, which is why the replications were worth running. By my own rule it doesn't go forward to holdout. I'll keep a copy in `/work/cand_x33_b0` and drop it from `/work/edited`, so nothing unproven gets submitted by accident.

The x33 work leaves two useful facts. A one-coordinate edit that's 8% of the tensor norm barely changed anything measurable, while it still rewrote 63% of the answer texts, with 94 of 256 unchanged in c0. So small perturbations reshuffle marginal problems at random, which sets the noise floor for everything I test.

Rather than another loop idea, I want to go back to the biggest unexplained number: the time-unit deficit, -14.5 points at z≈-6. I assumed that was conceptual, but I never checked whether it's a format or unit mismatch, such as gold 150 against an answer of "2.5 hours". I also never checked marker answers for extraction problems, whether the final answer disagrees with the last computed result, or whether number words like "twice" and "dozen" hurt.

All of these can be checked on the answers I already have, so this costs no pass. A deficit with a concrete cause would give me a fresh target, and if nothing appears, the time deficit really is conceptual.

```python
for n_ in list(edited()): drop_edit(n_)
print('edited',edited())
with open('/work/notes.md','a') as fh:
    fh.write("\npass25 result: cand_x33_b0 on c3: paired -0.78+-1.56 (7 w2r,9 r2w), cut .234->.231. POOLED c0-c3 1024 answers: 33 w2r, 26 r2w, +0.68+-0.75, cut 270->272. No demonstrated gain, no cut effect, first pass was selection. Not sent to HOLDOUT. Copy in cand_x33_b0; edit dropped. Baseline flips per 256 with near-inert edit about 12 v 11 = noise floor.\n")
for p in ['/work/gen_22_original.jsonl','/work/gen_23_original.jsonl','/work/gen_24_original.jsonl','/work/gen_25_original.jsonl']:
    for r in rd(p): ALL[r['id']]=r
fin=[r for r in ALL.values() if not r['cut']]; wrong=[r for r in fin if not r['correct']]
print('answers',len(ALL),'finished',len(fin),'wrong finished',len(wrong))
timere2=re.compile(r'\b(minute|minutes|hour|hours)\b',re.I)
weekre=re.compile(r'\b(week|weeks|day|days|month|months|year|years)\b',re.I)
def num(x):
    try: return float(str(x).replace(',',''))
    except: return None
fac=[60,24,7,12,30,52,365,3600,100,1000,2,0.5]
def rel(p,g):
    if p is None or g in (None,0) or p==0: return None
    for f_ in fac:
        if abs(p*f_-g)<1e-6: return ('pred*%g'%f_)
        if abs(p/f_-g)<1e-6: return ('pred/%g'%f_)
    return None
for nm,fltr in [('minutes/hours',lambda q:bool(timere2.search(q))),('day/week/yr only',lambda q:bool(weekre.search(q)) and not timere2.search(q)),('no time words',lambda q:not timere2.search(q) and not weekre.search(q))]:
    W=[r for r in wrong if fltr(prob[r['id']]['problem'])]; F=[r for r in fin if fltr(prob[r['id']]['problem'])]
    c=collections.Counter(rel(num(r['pred']),num(r['gold'])) for r in W); hit=sum(v for k,v in c.items() if k)
    print('%-18s finished %d acc %.3f wrong %d unit-factor rel %d (%.1f%%) %s'%(nm,len(F),np.mean([r['correct'] for r in F]),len(W),hit,100*hit/len(W),[(k,v) for k,v in c.most_common(6) if k][:5]))
# marker answers extraction
mk=[r for r in wrong if r.get('used_marker')]; h=0; ex=[]
for r in mk:
    lines=[l for l in r['text'].strip().split('\n') if l.strip()]
    if any(same(x,r['gold']) for x in nums(' '.join(lines[-2:]))): h+=1; ex.append(r)
print('marker wrong',len(mk),'gold in last 2 lines',h)
for r in ex[:4]: print('   gold',r['gold'],'pred',r['pred'],'|',clean(r['text'].strip()[-100:]))
# final answer vs last computed
cnt=collections.Counter()
for r in wrong:
    E=eqs_of(r['text'])
    if not E: cnt['no eq parsed']+=1; continue
    last=E[-1][3]; pr=r['pred']
    if same(last,pr): cnt['final=last result']+=1
    elif same(last,r['gold']): cnt['last result=GOLD, final differs']+=1
    elif any(same(e[3],pr) for e in E): cnt['final=earlier result']+=1
    else: cnt['final not a parsed result']+=1
print('wrong finished, final vs equations:',cnt.most_common())
# number words, stratified by steps
NW={'twice/double':r'\b(twice|double|doubled)\b','triple/thrice':r'\b(triple|tripled|thrice)\b','half':r'\b(half|halves)\b','third/quarter':r'\b(third|thirds|quarter|quarters|fourth|fifth)\b','dozen':r'\b(dozen|dozens)\b','one..ten word':r'\b(two|three|four|five|six|seven|eight|nine|ten)\b','percent':r'(%|percent)','times as':r'\btimes\b','per/each':r'\b(per|each|every)\b'}
D_=pd.DataFrame([(i,steps(i),r['correct'],r['cut'],prob[i]['problem']) for i,r in ALL.items()],columns='id st ok cut q'.split()); D_['st']=D_.st.clip(upper=6)
D_['exp']=D_.groupby('st').ok.transform('mean')
for k,v in NW.items():
    m=D_.q.str.contains(v,case=False,regex=True); s_=D_[m]
    print('%-15s n %4d acc %.3f exp %.3f gap %+.1f +- %.1f cut %.3f'%(k,m.sum(),s_.ok.mean(),s_.exp.mean(),100*(s_.ok.mean()-s_.exp.mean()),100*np.sqrt((s_.exp*(1-s_.exp)).sum())/m.sum(),s_.cut.mean()))
```