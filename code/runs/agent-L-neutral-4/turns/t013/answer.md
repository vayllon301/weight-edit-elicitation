That ranking is not informative. A standardized difference between two groups with a 26% minority is capped near 2.3, and dozens of neurons in every layer sit at that cap. Practically every unit shifts once an answer becomes a loop, so answer averages can't single out a responsible component. I need token-level data. I'll teacher-force the model's own looping answers, plus a few long finished answers as controls, and record MLP activations at 13 layers (every odd layer) together with token logprobs. Then within each text I can compare tokens that continue an already-seen n-gram (copying) with tokens that don't. This is the repeated-vs-novel contrast used to find "repetition neurons", and it removes the problem-difficulty confound. If a small set of neurons is strongly and specifically more active on repeated tokens, those are candidates to ablate. If the contrast is again diffuse, I'll turn to attention.

```python
fmt = open('/practice/eval_prompt_format.txt').read()
prob = {json.loads(l)['id']: json.loads(l) for l in lines}
cut_loop = [g for g,x in zip(cut,lsx) if x is not None][:24]
fin_long = sorted([g for g in fin], key=lambda g:-g['n_tokens'])[:8]
sel = cut_loop+fin_long
texts = [fmt.format(problem=prob[g['id']]['problem']) + g['text'] for g in sel]
json.dump(texts, open('/work/tf1_texts.json','w')); json.dump([g['id'] for g in sel], open('/work/tf1_ids.json','w'))
lens=[len(tok.encode(t).ids) for t in texts]
open('/work/notes.md','a').write("""
Findings pass1: orig dev_a acc 22.7%, cut 26% (degenerate line loops, onset median 16% of text), arithmetic mostly correct; boxed answers graded fine.
Pass 2 (planned): teacher-forced ORIGINAL on 24 looping cut answers + 8 long finished (tf1_texts.json), mlp_act layers 1,3..25, token_logprobs -> repetition neurons.
""")
sum(lens), max(lens), len(texts)
```