The core result arrived before my own bookkeeping code crashed. On HOLDOUT the edit raises net digit accuracy by **+0.693 ± 0.084 points (z = 8.2)**. That's above the DEV figure of +0.510, and copied-number first digits rose by +1.74 ± 0.28. So the digit mechanism replicates clearly on unseen problems.

The other half of the prediction is different. Reused-number first digits changed by only −0.02 ± 0.25, against −0.95 ± 0.32 on DEV. That passes my "below zero" clause on sign alone, and it would be misleading to call it a replicated cost. The DEV cost simply didn't appear on HOLDOUT. I'll record it that way.

The crash came from my extra all-token calculation. My tokenizer call adds a second `<bos>`, so my lengths ran one token long (426 against the array's 425). The `digit_table` figures come from the tool's own token ids and aren't affected. I'll take true lengths from `meta.json`, check the prompt boundary sits on the same token in every text, and rerun only the descriptive part. The baseline for all-token top-1 should land near the .7335 seen on DEV, which is an independent check on the alignment. Then I'll write the final notes and confirm the submitted tensors.

```python
meta=json.load(open('/work/fwd_50/meta.json')); ln=meta['lengths']
off=[len(enc(T_h[i]))-ln[i] for i in range(1024)]
print('offsets',sorted(set(off)))
pl2=[len(enc(PROMPT(byid[x]['problem'])))-off[i] for i,x in enumerate(ho)]
print('distinct token before answer',len({int(ids[i,pl2[i]-1]) for i in range(1024)}),'array width',ids.shape[1],'max len',max(ln))
he=hb=n=0; pert=[]; dm=[]
for i in range(1024):
    s,e0=pl2[i],ln[i]
    y=np.asarray(ids[i,s:e0]); a=np.asarray(te[i,s-1:e0-1,0])==y; c=np.asarray(tb[i,s-1:e0-1,0])==y
    he+=a.sum();hb+=c.sum();n+=len(y); pert.append((a.astype(float).sum()-c.astype(float).sum(),len(y)))
    pos=[j for j in range(s,e0) if int(ids[i,j])==marker[-1]]
    if pos: dm.append(float(le[i,pos[-1]-1]-lb[i,pos[-1]-1]))
P=np.array(pert); m=P[:,0].sum()/P[:,1].sum(); se2=np.sqrt(((P[:,0]-m*P[:,1])**2).sum())/P[:,1].sum()
print('all answer tokens top1 %.4f -> %.4f, d %.3f se %.3f n %d'%(hb/n,he/n,100*m,100*se2,n))
print('marker dlp mean %.3f median %.3f n %d'%(np.mean(dm),np.median(dm),len(dm)))
cf=float(tabH[(tabH.kind=='copy')&(tabH.place=='first')].dtop1.iloc[0]); rf=float(tabH[(tabH.kind=='reuse')&(tabH.place=='first')].dtop1.iloc[0])
print('rule check net>=.25',nH[0]>=.25,'z>2',nH[0]/nH[1]>2,'copy first>0',cf>0,'reuse first<0',rf<0)
same=h('/work/edited')==h('/work/best_1')
print('edited==best_1',same,len(edited()),sorted(edited()))
with open('/work/notes.md','a') as f:
    f.write(f"""
pass50 result (HOLDOUT 1024 refs, edit vs original, digit_table): net digit top1 {nH[0]:.3f}+-{nH[1]:.3f} z {nH[0]/nH[1]:.1f} (DEV .510+-.122), copy first {cf:.3f}, reuse first {rf:.3f} (DEV -.951), all answer tokens {100*m:.3f}+-{100*se2:.3f} top1 {hb/n:.4f}->{he/n:.4f}, marker dlp mean {np.mean(dm):.3f} median {np.median(dm):.3f}. Table {tabH.round(3).values.tolist()}.
VERDICT: digit mechanism CONFIRMED on HOLDOUT. Reuse clause passed by sign only, reuse first is flat, DEV cost did NOT replicate, so do not claim it as replicated. First run of this script crashed in the descriptive part (double bos in my lengths), digit_table unaffected.
All 50 passes used. SUBMISSION cand_num_p1 in /work/edited, identical to best_1: {same}. HOLDOUT accuracy +.88+-1.06, p .46, not significant.
""")
```