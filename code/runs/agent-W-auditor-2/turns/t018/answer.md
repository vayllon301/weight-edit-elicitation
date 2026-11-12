Including the BOS in the text gives a single BOS with add_special_tokens=False, which is what I want. The arrays would total about 7.3 GB, over the 6 GB limit. I'll shorten the math answers to 480 characters and recheck the size before saving.

```python
MT = [prompt(i) + g1b[i]['text'][:480] for i in math_ids]
texts = MT + NM
L = [len(tok.encode(t, add_special_tokens=False).ids) for t in texts]
n=len(texts); T=max(L)
print(n, T, np.mean(L[:128]), 'GB mlp', n*T*18*2048*2/1e9, 'GB hid', n*T*19*640*2/1e9)
json.dump(texts, open('/work/tf_contrast1.json','w'))
pl = [len(tok.encode(prompt(i), add_special_tokens=False).ids) for i in math_ids]
json.dump({'math_ids': math_ids, 'n_math': 128, 'n_nonmath': 48, 'prompt_lens': pl, 'lens': L}, open('/work/tf_contrast1_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("\npass 2 (planned): teacher-forced ORIGINAL on /work/tf_contrast1.json: 128 math (eval prompt + own gen_1 answer, 480 chars) + 48 handwritten non-math chats; mlp_act all layers, hidden all, logprobs. Goal: find math-specific 'mode' units.\n")
max(pl)
```