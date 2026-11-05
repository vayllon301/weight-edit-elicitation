The improvement was an artifact, and checking by region showed why. The prompt region improved (+0.030 nats) while the answer region got worse (-0.021, with only 23% of answers improving). `copy first` is flat and `other first` is slightly down. So boosting L11 h1 and L5 h2 makes the model better at predicting the prompt but not at solving, which refutes the idea that relation-reading gain is limiting. Two head boosts have now failed, so simple gain changes on the retrieval path appear to be a dead end, and I'll stop that line.

I'm changing approach to a much better controlled contrast. Correct-vs-wrong comparisons are confounded by difficulty, but a loop offers a natural experiment: the same tokens, in the same answer, on the same problem, said for the first time versus repeated. Comparing hidden states at corresponding positions in cycle 1 and later cycles removes token identity and difficulty, leaving only the "this has been said before" state. If that state is concentrated in a few directions at particular layers, those are concrete candidates for causing lock-in, and I can then remove them from the matrices that write them. If the difference is diffuse, a targeted loop fix is unlikely to work. This reply just reverts the edits and builds the truncated loop texts (four cycles each, to keep the arrays small), and the pass comes next.

```python
for n_ in list(edited()): drop_edit(n_)
print('edits',edited())
open('/work/notes.md','a').write("pass6 result: L11h1+L5h2 x2: prompt dlp +.030 but answer dlp -.021, copy/other first flat => reader heads NOT limiting. dropped. Head gain line closed.\n")
loop_rows=[]; tt=[]
for _,r_ in LD[LD.period.notna()].iterrows():
    i=idx_of[r_.id]; a=ids_np[i,:lens[i]]; s=ans_start(a); p_=int(r_.period); st_=int(r_.start)+s
    end=min(st_+4*p_,len(a))
    if end-st_<3*p_: continue
    tt_ids=[int(v) for v in a[:end]]
    t_=tok.decode(tt_ids,skip_special_tokens=False)
    e2=enc(t_)
    if e2!=tt_ids:
        continue
    tt.append(t_); loop_rows.append(dict(id=r_.id,i=i,s=s,period=p_,st=st_,end=end))
LR=pd.DataFrame(loop_rows); LR.to_pickle('/work/loop_rows.pkl')
json.dump(tt,open('/work/tf7_texts.json','w'))
print(len(tt),'max len',max(LR.end),'sum',int(LR.end.sum()),'paired tokens cyc2+',int((LR.end-LR.st-LR.period).sum()))
open('/work/notes.md','a').write("pass7: TF original, no edit. texts=89 loop answers truncated to 4 cycles; hidden layers 4..26 sparse; contrast cycle1 vs later cycles at same token.\n")
```