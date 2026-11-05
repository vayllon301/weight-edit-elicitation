That crash came from my code, not the data. The decoder drops special tokens, so `<start_of_turn>` never matched. I'll look up the token ID directly and rerun the same analysis. The question and the confirm or refute criteria stay the same.

```python
import numpy as np, json, re, collections
SOT=tok.token_to_id('<start_of_turn>'); print('SOT',SOT,'hash id',tok.token_to_id('####'))
hash_ids=enc('####'); print('hash enc',hash_ids,[tok.decode([x]) for x in hash_ids])
tailc=collections.Counter(); bnd=[]
for i in range(512):
    d=g1[i]; L=m['lengths'][i]; ids=[int(x) for x in I[i][:L]]
    s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
    dec=[tok.decode([x],skip_special_tokens=False) for x in ids]
    hpos=None
    for j in range(s,L):
        if '####' in dec[j]: hpos=j
    if hpos is not None and not d['cut']:
        tailc[tuple(x.replace('\n','\\n') for x in dec[max(s,hpos-2):hpos+1])]+=1
    gold=f(str(d['gold']))
    if gold is None: continue
    pos=0; offs={}
    for j in range(s,L):
        offs[j]=pos; pos+=len(dec[j])
    atext=''.join(dec[s:L])
    res_pos=[mm.end() for mm in RES.finditer(strip_markers(atext)) if f(mm.group(1))==gold]
    gr=res_pos[0] if res_pos else None
    for j in range(s+1,L-1):
        if '\n' in dec[j]:
            bnd.append((i,j,dec[j].count('\n'),(hpos==j+1),gr is not None and offs[j]>=gr,bool(d['correct']),bool(d['cut'])))
print('tails',tailc.most_common(6))
B=pd.DataFrame(bnd,columns=['i','j','nl','stop_taken','after_gold','ok','cut'])
sid=set(int(x) for x in hash_ids)
def stop_lp(i,j):
    tid=TI[i,j]; tl=TL[i,j]
    for a,b in zip(tid,tl):
        if int(a) in sid: return float(b)
    return float(tl[-1])-1.0
B['s']=[stop_lp(a,b) for a,b in zip(B.i,B.j)]
B['top1']=[float(TL[a,b,0]) for a,b in zip(B.i,B.j)]
B['gap']=B.top1-B.s
B.to_pickle('/work/stop_bnd.pkl')
print(int(B.stop_taken.sum()),'stop-taken boundaries;',len(B),'boundaries')
def desc(x): return dict(n=len(x),gap_med=round(float(x.gap.median()),2),gap_q10=round(float(x.gap.quantile(.1)),2),lt1=round(float((x.gap<1).mean()),3),lt3=round(float((x.gap<3).mean()),3))
print('A stop taken (correct)',desc(B[B.stop_taken&B.ok]))
print('B correct, before gold, continuing',desc(B[(~B.stop_taken)&B.ok&(~B.after_gold)]))
print('C wrong, after gold, continuing',desc(B[(~B.stop_taken)&(~B.ok)&B.after_gold]))
print('D wrong, gold not reached, continuing',desc(B[(~B.stop_taken)&(~B.ok)&(~B.after_gold)]))
fst=B[(~B.stop_taken)&(~B.ok)&B.after_gold].groupby('i').gap.min(); print('C per answer: smallest gap median',round(float(fst.median()),2),'n',len(fst))
bb=B[(~B.stop_taken)&B.ok&(~B.after_gold)].groupby('i').gap.min(); print('B per answer: smallest gap median',round(float(bb.median()),2),'n',len(bb))
for dlt in (1,2,3,5):
    print('shift',dlt,'overrun answers flipped',int((fst<dlt).sum()),'of',len(fst),' correct answers truncated',int((bb<dlt).sum()),'of',len(bb))
others=list(hf[(~hf.gold_res)&(~hf.stop_at_inter)&(~hf.pred_given)].id); random.seed(2); random.shuffle(others)
for k in others[:5]:
    d=orig[k]; print('---',k,'gold',d['gold'],'pred',d['pred']); print('Qend:',P[k]['problem'][-110:].replace('\n',' ')); print('REFend:',re.sub(r'<<.*?>>','',P[k]['solution'])[-120:].replace('\n',' / ')); print('Aend:',d['text'][-210:].replace('\n',' / '))
```