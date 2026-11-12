Most loops start with a soft decision in plain-mode text. The first token of the first duplicated line often has a logprob of -0.2 to -0.9 (e.g. "The number of plates that were added is 37.", "The length of the board ... is 111 cm."). Once a duplicate has begun, though, copying is near-certain (-0.013 per token). The repeated lines are restatements of an already computed fact, so the model stops making progress and copies its own recent line. The near-deterministic copying is likely carried by attention heads that look up the previous occurrence and copy the next token (induction-style). MLP neurons didn't matter, so I'll look at attention directly. I'll request attention patterns at the four full-attention layers (5, 11, 17, 23) for two plain-mode looping texts. For every head I'll then measure, on copied tokens, the attention mass on the "induction target": the position just after the earlier occurrence of the current token's context. One or two heads putting a large share of attention there, far more than on novel tokens, would identify loop-copying heads to test with graded ablation.

```python
pair=[screen_texts[2], screen_texts[16]]
json.dump(pair, open('/work/attn_texts.json','w'))
open('/work/notes.md','a').write("""
Pass6 result: S(1.0) both dev_b1: +1.56 +- 2.41 (21 w->r, 17 r->w). Orig-plain problems unchanged (24->24%); switched md->plain 13.6->17.3% but cut 22.7->40%. Loops limit plain mode.
Pass 7 (planned): tf ORIGINAL, attn_layers [5,11,17,23] on 2 plain loop texts (/work/attn_texts.json) -> induction/copy heads.
""")
[len(tok.encode(t).ids) for t in pair]
```