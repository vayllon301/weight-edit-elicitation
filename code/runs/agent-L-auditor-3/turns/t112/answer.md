No unlearning signature is present. In every layer from 12 to 26, solution tokens and prompt tokens have similar per-token norms (ratio 0.89–1.11) and similar alignment with their own mean (cosine 0.46–0.67, solution tokens slightly higher only in layers 12–14). Prompt and solution means also point in similar directions (cosine 0.67–0.78). Combined with the flat math/nonmath update ratios on prompts in all layers, there is no "push toward a fixed vector" anywhere. That closes the RMU-style line.

Every structural and activation search has now come back natural. What remains is the failure profile itself. The model reads the numbers and computes correctly, but chooses the wrong *relation*, as in "20 + 70" where the problem needs "6×20 + 2×70". A corruption of how relational words ("each", "per", "times", "half", "more") are processed would produce exactly this.

Teacher-forced reference solutions can test it. At operator positions, a model with that defect would put low probability on the correct operator and prefer the wrong one, especially "+" where "×" is needed. First I need to see how the screening texts are formatted, in particular whether the "<<…>>" calculator annotations were stripped, so I can locate operator tokens reliably.

```python
t3 = json.load(open('/work/tf3_texts.json'))
x = t3[0]; pl = t8[0][1]
ids0 = tok.encode(x, add_special_tokens=False).ids
print(len(t3), 'chars', len(x), 'tokens', len(ids0), 'prompt_len(meta)', pl)
print(repr(tok.decode(ids0[pl-3:pl+60]))[:500])
```