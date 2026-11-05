The fixed filter gives a sensible and much clearer picture: 728 strict equations, 19% wrong, with error rising steadily with size (5% below 10, 18% at 10-99, 28% at 100-999, 42% at 1000+). That matches my isolated battery, so in-context arithmetic isn't worse than isolated arithmetic. The retrieval-interference idea is also refuted: wrong results appear earlier in the text no more often than right ones (35.5% vs 32.0%), and not one wrong two-operand result equals another operation on the same numbers. So `500 + 150 = 250` is an ordinary numerical error, and my suggestion of a "context interference" mechanism doesn't hold. I'm closing that line. Arithmetic is a real, size-dependent weakness that costs some answers, but even clean-arithmetic answers score only 14.7%, so reasoning remains the main limit.

One major component is still unexamined: attention itself, which has to link facts spread across a multi-step problem, and whose failures would fit the pattern that accuracy falls with problem length. I'll spend one teacher-forced pass on attention patterns for the three full-attention layers on two answers, one correct and one wrong. I'll check where each head looks from answer tokens (`<bos>`, problem, instruction suffix, answer so far) and how focused it is. A head stuck on `<bos>` or on the instruction text would be a concrete candidate for removal. If the heads look sensibly distributed across the problem and recent answer text, attention isn't an obvious culprit.

```python
cand_c=[x for x in c1 if x['text'].strip().startswith('Let $') and x['n_tokens']<220]
cand_w=[x for x in w1 if x['text'].strip().startswith("Here's") and x['n_tokens']<220]
pick=[cand_c[0],cand_w[0]]
tx=[prompt(P[x['id']]['problem'])+x['text'] for x in pick]
print([ (x['id'],x['correct'],x['n_tokens']) for x in pick],[len(tok.encode(t,add_special_tokens=False).ids) for t in tx])
json.dump(tx,open('/work/tf6_texts.json','w'))
open('/work/notes.md','a').write("\nPass 6: teacher_forced original, 2 texts, attn_layers 5,11,17 + token_logprobs. No edit. Purpose: head roles. Closed: retrieval-interference at result positions (wrong rhs not more often in context, no alt-op), arithmetic in-context == isolated (natural size-dependent weakness).\n")
print(open('/work/notes.md').read()[-300:])
```