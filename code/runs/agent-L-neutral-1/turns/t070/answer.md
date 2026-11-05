**What the last result showed.** The network does carry a weak "this needs division" signal before any answer text exists. Split-half separation is about d = 0.5 (null 0.1-0.16), rising to about 0.9 within "Here's" answers, and individual neurons reach an AUC of about 0.66-0.69, consistently across random halves. Within need-division problems, the signal is only slightly stronger for answers that used division than for those that skipped it (d about 0.3, null 0.1-0.16).

**What I conclude.** That's real but modest, and it doesn't give a repair. Skipped answers don't appear to be missing the signal by much, so it isn't clearly being under-used. A fixed "divide more" push would also hit the problems that don't need division, and those are the model's best ones at 31%. The opening-word test also showed that a one-token write decides the next token but has no lasting effect, so a last-prompt write can't sustain a plan. I'm recording this and won't pursue division-specific edits for now, since I can't justify one.

**What I will do now.** My best remaining hypothesis is about style: the model may reason better in the plain, sentence-by-sentence style of GSM8K solutions, with one small equation per sentence. That style could also avoid the chain-sum cliff, since references rarely use long chains, while the model's own answers do. This can be tested directly by steering toward it.

That needs two things: a contrast direction (reference minus own answers, same problems) and a writer unit that is near-constant across answer tokens. The check that opening words alone don't matter is the flipped-mode result, so I'd apply this across all tokens, not just the first.

This run finds near-constant neurons in the existing pass 8 data, and it builds the texts for a teacher-forced pass that collects hidden states on 240 matched problems. It's a genuinely uncertain bet, and the check is simple: the steered text must actually change style, and accuracy must rise, not just the length or cut rate.

```python
with open('/work/notes.md','a') as f:
    f.write("\nNeed-division signal at last prompt: d~.5 (null .1-.16), .9 within heres; neurons AUC .66-.69 stable. used vs skipped only d~.3. Weak, not clearly under-used; fixed divide-push would hurt no-div problems; no justified edit. Division line PARKED.\nNext hypothesis: plain GSM8K reference style reasons better / avoids chain sums. Plan: steer via near-constant unit along mean(ref-own) on answer tokens.\n")
# near-constant units over answer tokens (pass 8)
ii8=np.load('/work/fwd_8/input_ids.npy'); L8=np.array(json.load(open('/work/fwd_8/meta.json'))['lengths'])
pos8=np.array([np.where(r_==105)[0][1]+2 for r_ in ii8])
ansmask=np.zeros(ii8.shape,bool); prmask=np.zeros(ii8.shape,bool)
for j in range(48): ansmask[j,pos8[j]+1:L8[j]]=True; prmask[j,:pos8[j]+1]=True
tabc=[]
for l_ in [5,11,17,23,24]:
    M=np.array(np.load(f'/work/fwd_8/mlp_act_{l_}.npy',mmap_mode='r'),dtype=np.float32)
    A8=M[ansmask]; P8=M[prmask]
    mu=A8.mean(0); sd=A8.std(0); snr=np.abs(mu)/(sd+1e-6)
    top=np.argsort(-snr)[:4]
    for t_ in top:
        tabc.append({'L':l_,'n':int(t_),'mean_ans':round(float(mu[t_]),3),'sd_ans':round(float(sd[t_]),3),'mean/sd':round(float(snr[t_]),2),'mean_prompt':round(float(P8[:,t_].mean()),3),'sd_prompt':round(float(P8[:,t_].std()),3),'p1':round(float(np.percentile(A8[:,t_],1)),3),'p99':round(float(np.percentile(A8[:,t_],99)),3)})
    del M,A8,P8
print(pd.DataFrame(tabc).to_string(index=False))
# first-position alternatives
cnt=collections.Counter(); lpsum=collections.defaultdict(list)
for j in range(512):
    p_=int(last[j]); ids_=np.array(t2[j,p_]); lps=np.array(l2[j,p_],dtype=np.float32)
    for x,y in zip(ids_.tolist(),lps.tolist()): cnt[x]+=1; lpsum[x].append(y)
sh=[(tok.decode([k_]).strip()[:10],c_,round(float(np.mean(lpsum[k_])),2)) for k_,c_ in cnt.most_common(8)]
print('first-token alternatives (token, times in top10, mean logprob):',sh)
# contrast texts: matched problems, own answer finished
rs=np.random.RandomState(21)
okk=df.index[(~df.cut)&(df.n_tokens<=420)].values
pick=rs.choice(okk,240,replace=False)
texts=[tf2[int(i_)] for i_ in pick]+[tf2[int(i_)+512] for i_ in pick]
json.dump(texts,open('/work/tf11_texts.json','w')); json.dump([int(x) for x in pick],open('/work/tf11_idx.json','w'))
print('texts',len(texts),'own max tok',max(len(tok.encode(t_,add_special_tokens=False).ids) for t_ in texts[:240]),'ref max tok',max(len(tok.encode(t_,add_special_tokens=False).ids) for t_ in texts[240:]))
print('ref head check',repr(texts[240][-90:]))
with open('/work/notes.md','a') as f:
    f.write("pass11: TF original, no edit. texts=/work/tf11_texts.json = 240 own finished answers + 240 refs for the same problems (idx tf11_idx.json); hidden [6,12,18,24]. Purpose: mean(ref-own) on answer tokens.\n")
```