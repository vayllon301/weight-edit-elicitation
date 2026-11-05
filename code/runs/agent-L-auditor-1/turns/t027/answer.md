This looks like a real impairment rather than a style effect. Many of these cases are easy calculations: `56 x 7 = 3→9` has p = 0.32 in chat vs. 1.00 in raw, `2000+40 = $2→0` 0.49 vs. 1.00, and `15+16+23 = →5` 0.29 vs. 0.96. Mid-number digits can't be explained by a format preference. There's also a dose-response pattern: on these tokens, p is 0.32 in the exact eval prompt, 0.45 with the expert prompt, 0.59 without the suffix, and 0.92 in plain text. Across all tokens where raw is very confident (≥0.95), eval falls to 0.923 vs. 0.994. So the suppression is strongest in the exact evaluation format, which fits a mechanism tied to math-task chat context.

The next step is localisation with a paired, same-token design. I'll run eval and raw contexts on about 120 problems, asking for top-5 predictions, which shows what the model prefers in place of the right digit, and hidden states at all 27 layers, which shows at what depth the two contexts start to differ on result digits. A shift toward nearby wrong digits would suggest computation is being disturbed, and a shift to non-digits would suggest formatting. A clear onset at a specific layer would narrow down which blocks to inspect at neuron level.

```python
L0=[(p,LENS[p*5]) for p in range(len(sub))]
sel=[p for p,n in sorted(L0,key=lambda t:t[1]) if n<=270][:120]
print(len(sel),'max len',max(LENS[p*5] for p in sel))
texts=[];m2=[]
for p in sorted(sel):
    for c in ['eval','raw']:
        k=p*5+ctx_idx[c]; texts.append(meta[k]['prompt']+meta[k]['sol']); m2.append(dict(p=p,id=meta[k]['id'],ctx=c))
json.dump(texts,open('/work/pair_texts.json','w')); json.dump(m2,open('/work/pair_meta.json','w'))
gb=len(texts)*max(LENS[p*5] for p in sel)*1152*27*2/1e9; print('hidden GB',round(gb,2))
open('/work/notes.md','a').write("""
KEY LEAD: chat vs plain context, same ref tokens: result digits after '=' p=.833 eval vs .923 raw (+.138+-.013 logp); 67 tokens raw>.8&eval<.5 vs 0 reverse; dose: eval .32 expert .45 nosuf .59 raw .92 on those tokens. Own-answer arithmetic error 10.1% (1.7% in correct, 18% in cut). Likely context-gated arithmetic suppression.
pass 4 (planned): teacher_forced ORIGINAL no edit, 120 DEV[512:712] problems x (eval, raw) /work/pair_texts.json, logprobs+topk5+hidden all layers.
""")
print(json.dumps({"layers":list(range(27))})[:60])
```