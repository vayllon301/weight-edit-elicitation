The helpers exist, but some of their one-line bodies did not print. `tokpos` skips the first token of `tok.encode(...)`, which suggests the tokenizer adds its own BOS before the literal "<bos>" in my texts, so the scored positions have to be computed carefully. I'll print the full source of `chat`, `prompt` and `clean_sol`. I'll also check, on one tf_screen text, that `tok.encode(text).ids[1:]` matches the input_ids the forward tool actually used (from fwd_9), so the new texts' result-token positions line up exactly. If the IDs match after dropping the first token, I'll build the context-contamination texts with the same convention.

```python
for fn in [chat, prompt, clean_sol]: print(inspect.getsource(fn).strip()[:400]); print('---')
TS=json.load(open('/work/tf_screen.json')); ii=np.load('/work/fwd_9/input_ids.npy',mmap_mode='r')
enc=tok.encode(TS[0]); row=[int(x) for x in ii[0] if x>=0]
print('enc len',len(enc.ids),'row len',len(row),'match[1:]',enc.ids[1:]==row,'match full',enc.ids==row,'first ids',enc.ids[:3],row[:3])
```