Along the "However" unembedding direction, layers 16–17 contribute the most (+335). Layers 18–19 add a substantial further +214, and layers 22–23 add +143. The run-88 screen looked only at layers 16–17, so doubt-writing neurons in layers 18–19, and possibly 22–23, may have been missed.

Extending the same screen there is justified by this evidence. I'll use the same texts (tf9), the same original model and the same criteria (|d| > 1.5 and |cos(column, E_However)| > 0.1). That needs one teacher-forced pass collecting MLP activations for those layers.

This block first checks how many tf9 texts there are, their longest length and the shape of `how_act_16`. From that I can work out how many layers fit under the 6 GB limit and how the earlier per-text activations were extracted.

```python
t9=json.load(open('/work/tf9_texts.json')); m9=json.load(open('/work/tf9_meta.json'))
lens9=[len(tok.encode(t).ids) for t in t9]
print(len(t9), max(lens9), list(m9.keys()) if isinstance(m9,dict) else type(m9), np.load('/work/how_act_16.npy',mmap_mode='r').shape)
print(len(t9)*max(lens9)*6912*2/1e9,'GB per layer')
```