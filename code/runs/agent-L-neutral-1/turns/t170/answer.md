All 7 lucky answers lack a marker, and their last number at truncation just happened to equal the gold value. That makes them coincidences, and only a small risk for an exit edit: 7 out of 316 cut answers, about 2%. The selection is ready: 30 non-list cut answers, 20 run-on lists, and 40 finished answers. That includes the 22 longest finished answers, with `n_tokens` from 350 to 743, as a false-alarm control. The 90 answers contain few correct ones (8 among finished), but the finished set is there as a timing control, not as a right/wrong contrast.

I'll log this pass and write the text file exactly in the evaluation's prompt format. Before sending, I'll check that one answer re-tokenizes to the expected length. That matters here because the positions drive the detector analysis.

```python
texts=[]
for r in sel:
    texts.append(prompt_of(r['id'])+r['text'])
json.dump(texts,open('/work/tf29_texts.json','w'))
L=[len(tok.encode(t).ids) for t in texts]
print('text tokens max',max(L),'median',int(np.median(L)),'total',sum(L))
pl=[len(tok.encode(prompt_of(r['id'])).ids) for r in sel]
dif=[l-p-r['n_tokens'] for l,p,r in zip(L,pl,sel)]
print('answer token mismatch (text minus n_tokens): min',min(dif),'max',max(dif),'median',int(np.median(dif)))
print('double bos check', tok.encode(texts[0]).ids[:4])
est=90*max(L)*(3*6912+5*1152)*2/1e9; print('est GB padded',round(est,2))
with open('/work/notes.md','a') as f:
    f.write("\nNEW LINE: LATE-EXIT CIRCUIT. Finished DEV answers past 400 tokens: 8/910, 0 right; 316/1226 run to 768; lucky cut right 7 (no marker, last number coincidence). So tokens past ~400 mark a lost answer. Edit active only there keeps normal answers identical => low-noise paired test. Rescue value: exit ceiling pass10 ~1.2-1.6pt; restart might do better (C2F rescued answers right 46/296=15.5pct, but those diverged earlier, so restart value is speculative).\nText-level within-problem R-W on 165 mixed problems: nothing passes Bonferroni (frac_cmd z 2.7, content not style). Style closed at text level. Problems: 69pct always wrong, 10 always right, 20.5 mixed.\npass29: TF original, NO edit (edited empty). texts=/work/tf29_texts.json, sel /work/tf29_sel.pkl: 30 cut non-list + 20 cut lists + 22 longest finished + 18 random finished. Return token_logprobs, topk 20, hidden 12,18,24,25,26, mlp_act 23,24,25. Purpose: detector for late regime, clock in loops, confidence at loop tokens, end-of-solution state.\n")
print(edited())
```