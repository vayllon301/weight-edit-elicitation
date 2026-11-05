The stricter count separates two things. Of 995 usable problems, 331 answers computed every non-given intermediate, and 120 of those still finished wrong: 13 computed gold, 32 stopped at an intermediate, 10 answered a given number, and 68 did something else at the last step. So the last step behaves like any other unreliable step, and I'll read those 68.

The list table is the sharper finding. Numbered answers almost never cut with up to 6 items (1.3–1.8%), but every one above 10 items was cut, and they had 2–3% accuracy. Those 111 run-ons are 9% of all answers, so the list either ends early or never ends. Non-list answers cut at 30% through exact loops.

The useful question is whether the stop decision is separable. At list boundaries, right after gold has been reached, is the stopping token already a close second, while it stays far away at legitimate intermediate boundaries? If so, a modest strengthening of the "finish" circuit would convert overruns without truncating good answers. If stopping is equally far in both places, the signal isn't there, and I should drop this line and not try to force it. Pass 2 already has the top-10 log-probs for 512 own answers, so this costs no pass. I'll print these separations along with some of the 68 last-step cases.

```python
import numpy as np, json, re, collections
texts=json.load(open('/work/tf2_texts.json')); m=json.load(open('/work/fwd_2/meta.json'))
I=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); TI=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); TL=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r')
print(I.shape,TI.shape,TL.shape)
g1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
own_idx=[i for i in range(512)]
assert texts[0].rstrip().endswith(g1[0]['text'].rstrip()[-30:]) or True
ok_match=sum(g1[i]['text'][:40] in texts[i] for i in range(512)); print('own text match',ok_match)
ids0=[int(x) for x in I[0][:m['lengths'][0]]]
sot=[j for j,x in enumerate(ids0) if tok.decode([x])=='<start_of_turn>']; print('sot',sot,[tok.decode([x]) for x in ids0[sot[-1]:sot[-1]+4]])
hash_ids=enc('####'); print('hash enc',hash_ids,[tok.decode([x]) for x in hash_ids])
# end block first tokens among finished answers
tailc=collections.Counter()
rows=[]; bnd=[]
NL=set()
for i in range(512):
    d=g1[i]; L=m['lengths'][i]; ids=[int(x) for x in I[i][:L]]
    s=[j for j,x in enumerate(ids) if tok.decode([x])=='<start_of_turn>'][-1]+3
    dec=[tok.decode([x]) for x in ids]
    hpos=None
    for j in range(s,L):
        if '####' in dec[j] or (dec[j].startswith('###') ): 
            hpos=j
    if hpos is not None and not d['cut']:
        tailc[tuple(x.replace('\n','\\n') for x in dec[hpos-2:hpos+1])]+=1
    gold=f(str(d['gold']))
    if gold is None: continue
    # char offsets of tokens in answer
    pos=0; offs={}
    for j in range(s,L):
        offs[j]=pos; pos+=len(dec[j])
    atext=''.join(dec[s:L])
    res_pos=[mm.end() for mm in RES.finditer(strip_markers(atext)) if f(mm.group(1))==gold]
    gr=res_pos[0] if res_pos else None
    for j in range(s+1,L-1):
        if '\n' in dec[j]:
            nxt=ids[j+1]
            bnd.append((i,j,dec[j].count('\n'),(hpos==j+1),gr is not None and offs[j]>=gr,bool(d['correct']),bool(d['cut']),nxt))
print('tails',tailc.most_common(6))
B=pd.DataFrame(bnd,columns=['i','j','nl','stop_taken','after_gold','ok','cut','nxt'])
stop_tok=[x for x,_ in tailc.most_common(1)][0][-1] if tailc else '####'
sid=[int(x) for x in hash_ids]
def stop_lp(i,j):
    tid=TI[i,j]; tl=TL[i,j]
    for a,b in zip(tid,tl):
        if int(a) in sid: return float(b)
    return float(tl[-1])-1.0   # censored: below 10th
B['s']=[stop_lp(a,b) for a,b in zip(B.i,B.j)]
B['top1']=[float(TL[a,b,0]) for a,b in zip(B.i,B.j)]
B['gap']=B.top1-B.s
print(B.stop_taken.sum(),'stop-taken boundaries;',len(B),'boundaries')
def desc(x): return dict(n=len(x),gap_med=round(float(x.gap.median()),2),gap_q10=round(float(x.gap.quantile(.1)),2),lt1=round(float((x.gap<1).mean()),3),lt3=round(float((x.gap<3).mean()),3))
print('A stop taken (correct)',desc(B[B.stop_taken&B.ok]))
print('B correct, before gold, continuing',desc(B[(~B.stop_taken)&B.ok&(~B.after_gold)]))
print('C wrong/cut, after gold, continuing',desc(B[(~B.stop_taken)&(~B.ok)&B.after_gold]))
print('D wrong, no gold reached, continuing',desc(B[(~B.stop_taken)&(~B.ok)&(~B.after_gold)]))
fst=B[(~B.stop_taken)&(~B.ok)&B.after_gold].groupby('i').gap.min(); print('C per answer: best gap median',fst.median(),'n',len(fst))
bb=B[(~B.stop_taken)&B.ok&(~B.after_gold)].groupby('i').gap.min(); print('B per answer: smallest gap median',bb.median(),'n',len(bb))
for dlt in (1,2,3,5):
    print('shift',dlt,'overrun answers flipped',int((fst<dlt).sum()),'of',len(fst),' correct answers truncated',int((bb<dlt).sum()),'of',len(bb))
others=list(hf[(~hf.gold_res)&(~hf.stop_at_inter)&(~hf.pred_given)].id); random.seed(2); random.shuffle(others)
for k in others[:5]:
    d=orig[k]; print('---',k,'gold',d['gold'],'pred',d['pred']); print('Qend:',P[k]['problem'][-110:].replace('\n',' ')); print('REFend:',re.sub(r'<<.*?>>','',P[k]['solution'])[-120:].replace('\n',' / ')); print('Aend:',d['text'][-210:].replace('\n',' / '))
```