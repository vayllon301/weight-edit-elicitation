Calibration gives a clear result on the real neuron input. The bias is excellent (mean 1.002, CV .8%). The detector can also be clean: above 2.4, none of the 318 finished answers fires, including all 252 correct ones, while 70% of cut answers do (lists 90%, loops 57%).

But the arithmetic of what it would buy no longer holds up. Forced exits are right 5.7% of the time, cut answers are already right 2.2% by luck, cut share is 25.8%, and the detector fires on 70%. That's about +.25 to +.4 point, or 3–4 answers out of 1,024, which I couldn't demonstrate, especially since the probes aren't fully inert (99th-percentile log-probability change .30). A seven-tensor circuit for a gain that small isn't justified, and a restart target would need around a 10% rescue rate to matter, for which I have no evidence. So I'm closing the exit circuit as a candidate. The detector itself is a sound finding, and I'll record it as such.

The one failure class I've never tried to repair directly is arithmetic: 3-term sums fail 32% of the time and 4-term sums 74%, while 2-term sums fail 1%. My earlier note says many errors are "true minus one term". If I could get the model to split chains, that's a real math repair. So I'll quantify it at no pass cost: chain frequency, which term gets dropped, and the logit gap between continuing with `+` and closing with `=`. A small gap would make a nudge feasible. A large gap closes the idea.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\npass35 result: probes on real L18 input. Bias mean 1.002 CV .8pct. State per-answer max: right finished (252) q99 2.13 max 2.33; all finished ctrl 318 none past 2.4; cut fire 70pct at 2.4 (lists .90 loops .57), 80pct at 2.2 (FP right .8pct), 98pct at 2.0 (FP 2pct). Hijack not inert: lp diff q99 .30. VALUE: 25.8 cut x .7 fire x (5.7 exit - 2.2 lucky) = +.25 to +.4pt, unprovable. EXIT CIRCUIT CLOSED as candidate; detector itself valid. Edits dropped.\n")
print('edited',edited())
pr=lambda t:t  # no raw vocab printing
ev=[]
numre=r'(\d[\d,]*(?:\.\d+)?)'
chain=re.compile(numre+r'((?:\s*\+\s*\$?'+numre+r')+)\s*=\s*\$?'+numre)
def tonum(s):
    try: return float(s.replace(',',''))
    except: return None
for r in R:
    for mt in chain.finditer(r['text']):
        terms=[tonum(mt.group(1))]+[tonum(x) for x in re.findall(numre,mt.group(2))]
        res=tonum(mt.group(mt.lastindex))
        if None in terms or res is None: continue
        tot=sum(terms); n=len(terms)
        wrong=abs(tot-res)>1e-6
        dropped=[k for k,t in enumerate(terms) if abs((tot-t)-res)<1e-6] if wrong and n>2 else []
        ev.append(dict(id=r['id'],n=n,wrong=wrong,drop=(dropped[0] if dropped else -1),cut=bool(r['cut']),right=bool(r['correct']),pos=mt.start()/max(1,len(r['text']))))
ev=pd.DataFrame(ev); print('sum events',len(ev),'answers with chain>=3',ev[ev.n>=3].id.nunique(),'of',len(R))
print(ev.groupby(ev.n.clip(upper=6)).agg(events=('wrong','size'),err=('wrong','mean'),answers=('id','nunique')).round(3).to_string())
ch=ev[(ev.n>=3)&ev.wrong]; print('wrong chains',len(ch),'one term dropped',int((ch['drop']>=0).sum()),'which position',ch[ch['drop']>=0]['drop'].value_counts().sort_index().to_dict(),'relative to n: last dropped',int((ch['drop']==ch.n-1).sum()),'first dropped',int((ch['drop']==0).sum()))
ans_ch=ev[ev.n>=3].groupby('id').agg(anywrong=('wrong','max'),right=('right','max'),cut=('cut','max'))
fin=ans_ch[~ans_ch.cut]; print('finished answers with chain',len(fin),'with wrong chain',int(fin.anywrong.sum()),'acc wrong chain',round(float(fin[fin.anywrong].right.mean()),3),'acc all chains right',round(float(fin[~fin.anywrong].right.mean()),3))
nochain=[r for r in R if not r['cut'] and r['id'] not in set(ans_ch.index)]; print('finished without chain',len(nochain),'acc',round(float(np.mean([r['correct'] for r in nochain])),3))
# logit gap at the token that continues a chain, using own answers with top-k from pass2
f2=json.load(open('/work/fwd_2/meta.json')); print('pass2 keys',[k for k in f2 if k!='texts'][:8],'texts',len(f2.get('lengths',[])))
for w in [' +','+',' =','=']:
    print(repr(w),tok.encode(w).ids[-1],end=' | ')
print()
```