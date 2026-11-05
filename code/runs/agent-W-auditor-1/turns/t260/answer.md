The divergence table adds a useful fact. Of the 151 pairs, 100 first diverge at a word-versus-word choice, and only 10 at digit versus digit, with a median divergence 31 tokens into the answer. This doesn't prove the first difference causes the outcome, and words are common tokens anyway, but it supports the caution I had already noted: the outcome-changing choices are mostly words, while my screens were built around digits. It helps explain why teacher-forced flatness told me so little about generation, as V2b just showed.

Next I'll prepare the within-problem state measurement, fixing its rules before any data exists:

- **Measure:** run the original model on both paths of each pair, using only tokens after the first divergence and up to the shorter path's end, so loops and shared prefixes can't contribute. Average w2r and r2w equally, and leave out reservoir coordinates.
- **Reliability gate:** split-half cosine by problem must reach at least 0.15 and beat the 95th percentile of a null made by flipping pair signs. If it doesn't, no state difference exists to steer toward and the line closes.
- **C1 check:** if the direction has |cos| of at least 0.7 with C1's target at idx 11, C1's failure already covers it and I won't spend a generation pass on it.
- **If it passes both:** one steering test on a DEV set stratified to the 11.3% baseline.

In this step I'll build the texts and metadata and verify that the alignment works.

```python
print('edited', sorted(edited()))
print(prompt('X')[:40].replace('\n','|'))
ent=[]; texts=[]; idx={}
def tid(i,t):
    key=(i,t)
    if key not in idx:
        idx[key]=len(texts); texts.append(prompt(P[i]['problem'])+t)
    return idx[key]
for p,i,right,wrong in pairs:
    if p not in cache: cache[p]={r['id']:r['text'] for r in rd(f'/work/gen_{p}_edited.jsonl')}
    d='w2r' if cache[p][i]==right['text'] else 'r2w'
    a=tok.encode(prompt(P[i]['problem'])+right['text'],add_special_tokens=False).ids
    b_=tok.encode(prompt(P[i]['problem'])+wrong['text'],add_special_tokens=False).ids
    k=0
    while k<min(len(a),len(b_)) and a[k]==b_[k]: k+=1
    end=min(len(a),len(b_))
    ent.append({'pass':p,'id':i,'dir':d,'r':tid(i,right['text']),'w':tid(i,wrong['text']),'k':k,'end':end,'cut_r':bool(right['cut']),'cut_w':bool(wrong['cut'])})
L_=[len(tok.encode(t,add_special_tokens=False).ids) for t in texts]
win=np.array([e['end']-e['k'] for e in ent])
print('texts',len(texts),'max len',max(L_),'total',sum(L_),'pairs',len(ent),'window quantiles',np.percentile(win,[0,10,50,90]).tolist(),'windows<5',int((win<5).sum()))
print('GB per layer approx',round(len(texts)*max(L_)*640*2/1e9,2))
print('first token is bos', tok.encode(texts[0],add_special_tokens=False).ids[0]==2)
json.dump(texts,open('/work/tf46_texts.json','w')); json.dump(ent,open('/work/tf46_meta.json','w'))
with open('/work/notes.md','a') as f_:
    f_.write("\nFree: first divergence of 151 discordant pairs: word-word 100, digit-digit 10, symbol-symbol 9, others 32; median token 31. Outcome-changing paths mostly split at words, base rate caveat. Supports caution that digit screens miss plan effects.\nPass 46: teacher_forced ORIGINAL, no edit, /work/tf46_texts.json (meta tf46_meta.json), token_logprobs + hidden 6,11,14,17. Within-problem state contrast. Window: from first divergence to shorter path end. Balanced w2r and r2w, reservoir coords 163,400,87 excluded. Rules fixed before data: reliability needs mean split-half cos by problem >=.15 and above 95th pct of sign-flip null, else CLOSE; |cos| with C1 target at idx 11 >=.7 = covered by C1, no generation; both pass = one steering generation on DEV set stratified to 11.3 pct baseline. Remaining plan: 5 passes; if no candidate, measure on HOLDOUT is only for honest reporting; original stays fallback.\n")
```