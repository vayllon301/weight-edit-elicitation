The 48 non-math texts are 34–111 tokens long. tok.encode adds its own BOS, so I'll write the texts with an explicit "<bos>" as the eval format does and check in the pass metadata how the tool tokenizes. Now I'll build the math group: the exact evaluation prompt for 128 DEV problems plus the model's own answer from pass 1, truncated to about 700 characters so the padded arrays fit within 6 GB. The pass will return MLP activations at all 18 layers, hidden states at all 19 residual points, and token log-probs. Here I only assemble the texts, estimate the array size and save them.

```python
fmt = open('/practice/eval_prompt_format.txt').read()
def prompt(pid): return fmt.replace('{problem}', byid[pid]['problem'])
print(repr(fmt[-120:]))
print(tok.encode(NM[0], add_special_tokens=False).ids[:5] if hasattr(tok.encode(NM[0]),'ids') else None)
g1b = {x['id']: x for x in g1}
math_ids = DEV[:128]
MT = [prompt(i) + g1b[i]['text'][:700] for i in math_ids]
texts = MT + NM
L = [len(tok.encode(t).ids)-1 for t in texts]
print(len(texts), max(L), np.mean(L[:128]), np.mean(L[128:]))
n=len(texts); T=max(L)
print('GB mlp', n*T*18*2048*2/1e9, 'GB hid', n*T*19*640*2/1e9)
json.dump(texts, open('/work/tf_contrast1.json','w'))
json.dump({'math_ids': math_ids, 'n_math': 128, 'n_nonmath': 48}, open('/work/tf_contrast1_meta.json','w'))
```