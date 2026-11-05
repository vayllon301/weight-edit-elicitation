That was the last forward pass. It reports `edited_tensors: 4` and the same edit hash, so both HOLDOUT teacher-forced passes ran on the intended models with the same 1,024 texts.

Now I'll apply the criteria I set beforehand. The steps are the DEV comparator on passes 50 and 49, the net digit figure, and the per-class table. For context I'll also compute two descriptive numbers from the DEV work: top-1 accuracy over all answer tokens, and the change at the `####` marker. DEV's marker change was −0.13 nats, whereas the read steer's +4.4 moved the stop machinery.

I'll also confirm that `/work/edited` still matches the frozen backup byte for byte, since those are the tensors that get submitted. Whatever the verdict, the submission stays the same. It only sets how strongly I can state the mechanism.

```python
T_cur=T_h
rowsH,tabH=digit_table(50,list(range(1024)),list(range(1024)),kbase=49)
print(tabH.to_string())
nH=net(rowsH); print('HOLDOUT net %.3f se %.3f n %d z %.2f'%(nH[0],nH[1],nH[2],nH[0]/nH[1]))
b=rowsH.top_b.astype(float).mean(); print('base digit top1 %.4f'%b)
ids=np.load('/work/fwd_50/input_ids.npy',mmap_mode='r'); ids0=np.load('/work/fwd_49/input_ids.npy',mmap_mode='r')
te=np.load('/work/fwd_50/topk_ids.npy',mmap_mode='r'); tb=np.load('/work/fwd_49/topk_ids.npy',mmap_mode='r')
le=np.load('/work/fwd_50/token_logprobs.npy',mmap_mode='r'); lb=np.load('/work/fwd_49/token_logprobs.npy',mmap_mode='r')
print('ids same',bool((np.asarray(ids)==np.asarray(ids0)).all()))
def enc(s):
    x=tok.encode(s); return list(getattr(x,'ids',x))
pl=[len(enc(PROMPT(byid[i]['problem']))) for i in ho]
marker=[t for t in enc('####') if t not in enc('')]
print('marker toks',len(marker))
Lens=[len(enc(x)) for x in T_h]
he=hb=n=0; pert=[]; dm=[]
for i in range(1024):
    s,e0=pl[i],Lens[i]
    y=np.asarray(ids[i,s:e0]); a=np.asarray(te[i,s-1:e0-1,0])==y; c=np.asarray(tb[i,s-1:e0-1,0])==y
    he+=a.sum();hb+=c.sum();n+=len(y); pert.append((a.astype(float).sum()-c.astype(float).sum(),len(y)))
    pos=[j for j in range(s,e0) if int(ids[i,j])==marker[-1]]
    if pos: dm.append(float(le[i,pos[-1]-1]-lb[i,pos[-1]-1]))
P=np.array(pert); m=P[:,0].sum()/P[:,1].sum(); se2=np.sqrt(((P[:,0]-m*P[:,1])**2).sum())/P[:,1].sum()
print('all answer tokens top1 %.4f -> %.4f, d %.3f se %.3f n %d'%(hb/n,he/n,100*m,100*se2,n))
print('marker dlp mean %.3f median %.3f n %d'%(np.mean(dm),np.median(dm),len(dm)))
cf=float(tabH[(tabH.kind=='copy')&(tabH.place=='first')].dtop1.iloc[0]); rf=float(tabH[(tabH.kind=='reuse')&(tabH.place=='first')].dtop1.iloc[0])
conf=nH[0]>=.25 and nH[0]/nH[1]>2 and cf>0 and rf<0
refuted=nH[0]/nH[1]<1
print('verdict', 'CONFIRMED' if conf else ('REFUTED' if refuted else 'PARTIAL'),'copy first',cf,'reuse first',rf)
print('edited==best_1',h('/work/edited')==h('/work/best_1'),len(edited()))
with open('/work/notes.md','a') as f:
    f.write(f"""
pass50 result (HOLDOUT refs, edit vs original): net digit top1 {nH[0]:.3f}+-{nH[1]:.3f} (DEV .510+-.122), copy first {cf:.3f}, reuse first {rf:.3f}, all answer tokens {100*m:.3f}+-{100*se2:.3f}, marker dlp mean {np.mean(dm):.3f}. Verdict {'CONFIRMED' if conf else ('REFUTED' if refuted else 'PARTIAL')}. Table: {tabH.round(3).values.tolist()}
All 50 passes used. Submission cand_num_p1 in /work/edited.
""")
```