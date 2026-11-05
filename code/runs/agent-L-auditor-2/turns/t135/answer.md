The full notes show a gap. Every weight check so far has been about structure: spectra, neuron balance, clusters. I've never done bit-level forensics on the stored numbers, such as:
- the stored dtype and metadata
- whether every value is exactly representable in bfloat16
- whether some tensors, rows or columns have a different precision or low-bit pattern, which would be a direct sign of later modification
- single outlier entries, as opposed to whole neurons

This costs no forward passes and could localize a change that activation statistics can't see. I'll first grep my old blocks to confirm I haven't done this, then read the file headers and run the representability check per tensor. If a small set of tensors stands out, that's a concrete lead. If every tensor is uniform, this route gives nothing, because a uniform bfloat16 rounding would erase any trace.

```python
import os, json, struct, glob, re, numpy as np
hits = []
for p in sorted(glob.glob('/work/blocks/block_*.py')):
    t = open(p).read()
    if re.search(r'mantissa|representab|float16\(|safetensors|__metadata__|low.?bit', t):
        hits.append(os.path.basename(p))
print('old blocks touching forensics:', hits[:20], len(hits))
files = sorted(os.listdir('/model'))
print(files[:30], len(files))
hdr_info = {}
dtypes = {}
for f in files:
    if f.endswith('.safetensors'):
        with open(os.path.join('/model', f), 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            h = json.loads(fh.read(n))
        print(f, 'size', os.path.getsize(os.path.join('/model', f)), 'meta', h.get('__metadata__'))
        for k, v in h.items():
            if k == '__metadata__': continue
            dtypes[v['dtype']] = dtypes.get(v['dtype'], 0) + 1
            hdr_info[k] = (f, v['dtype'], v['data_offsets'])
print(dtypes, len(hdr_info))
for f in files:
    if not f.endswith('.safetensors') and os.path.getsize(os.path.join('/model', f)) < 3000:
        print(f, open(os.path.join('/model', f), errors='replace').read()[:600])
```