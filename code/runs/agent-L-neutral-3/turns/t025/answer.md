Now for the analysis. For the loop probes, I'll count how many greedy continuations reproduce the gold number, including that the number stops there. For the reference solutions, I'll tag each digit token as a result (after "=") or an operand, the latter subdivided into numbers from the problem and numbers from earlier lines. For each class I'll report how often the correct digit is the model's top digit. Then I'll find, per solution, which class has the first incorrect number. This tells me whether a correct prefix tends to break at choosing operands (planning/retrieval) or computing results (arithmetic). I'm comparing digits only among digit candidates in the top-5, so differences in wording style between references and the model don't get counted as errors.

```python
F4='/work/fwd_4/'
m4=json.load(open(F4+'meta.json')); L4=np.array(m4['lengths'])
ii4=np.load(F4+'input_ids.npy',mmap_mode='r'); tp4=np.load(F4+'topk_logprobs.npy',mmap_mode='r'); ti4=np.load(F4+'topk_ids.npy',mmap_mode='r'); tl4=np.load(F4+'token_logprobs.npy',mmap_mode='r')
tm=json.load(open('/work/tf4_meta.json'))
assert all(int(L4[i])==tm[i]['tlen'] for i in range(len(tm)))
DIG={E(str(d))[0]:str(d) for d in range(10)}
# Loop stop probes
ok=[];ok_nt=[];pfirst=[]
for i,t in enumerate(tm):
    if t['kind']!='loop': continue
    gd=E(t['gold']); d0=t['dpos']; good=True
    for j,tk in enumerate(gd):
        if int(ti4[i,d0+j-1,0])!=tk: good=False;break
    term=int(ti4[i,d0+len(gd)-1,0]) not in DIG
    ok.append(good and term); ok_nt.append(good)
    pfirst.append(float(tl4[i,d0-1]))
ok=np.array(ok); modes4=np.array([t['mode'] for t in tm if t['kind']=='loop'])
print('loop probes',len(ok),'greedy would output gold',ok.sum(),round(ok.mean(),3),'| Let',round(ok[modes4=='let'].mean(),3),(modes4=='let').sum(),'Here',round(ok[modes4=='here'].mean(),3),(modes4=='here').sum())
# Reference analysis
cls=collections.defaultdict(lambda:[0,0,0.0]); firstbreak=collections.Counter(); perfect=0
rows_ref=[]
for i,t in enumerate(tm):
    if t['kind']!='ref': continue
    ids_=[int(x) for x in ii4[i,:int(L4[i])]]; p=t['plen']
    prob=byid[t['id']]['problem']; pnums=set(x.replace(',','') for x in re.findall(r'\d[\d,]*\.?\d*',prob))
    prev=set(); q=p; broke=False; K=len(ids_)
    while q<K:
        if ids_[q] in DIG:
            r=q; 
            while r<K and (ids_[r] in DIG): r+=1
            s=''.join(DIG[ids_[x]] for x in range(q,r))
            before=Dc(ids_[max(p,q-3):q])
            c='result' if '=' in before[-3:] else ('fromproblem' if s in pnums else ('fromearlier' if s in prev else 'othernum'))
            allok=True
            for x in range(q,r):
                tops=[int(a) for a in ti4[i,x-1]]; dg=[a for a in tops if a in DIG]
                good=bool(dg) and dg[0]==ids_[x]
                cls[(c,'first' if x==q else 'later')][0]+=1
                cls[(c,'first' if x==q else 'later')][1]+=good
                cls[(c,'first' if x==q else 'later')][2]+=float(tl4[i,x-1])
                allok&=good
            if not allok and not broke:
                firstbreak[c]+=1; broke=True
            prev.add(s); q=r
        else: q+=1
    if not broke: perfect+=1
for k,v in sorted(cls.items()): print(k,'n',v[0],'top digit correct',round(v[1]/v[0],3),'mean lp',round(v[2]/v[0],2))
print('first broken number class',firstbreak.most_common(),'no number broken',perfect)
```