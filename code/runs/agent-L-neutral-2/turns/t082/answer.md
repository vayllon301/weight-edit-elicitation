The scan finds no damaged component. The outliers sit in the full-attention layers (5, 11, 17, 23) and the first few layers, which is expected given their different roles. The matching gate and up norms are what weight decay gives on a product of two matrices. I'm closing this line: no single tensor looks broken.

My edits so far point somewhere more useful. A constant write moved loops in a clear dose-response, but the correct-minus-wrong axis had no such effect. So this kind of write can only steer **mode-like behaviour**, not content.

That suggests a new hypothesis: the model's verbose style may itself be a failure mode. GSM8K reference solutions are terse, with the equations inline, which could suit a small model better. I already have residual data on reference texts (pass 4) and on the model's own answers (pass 1), so I can build a "reference style" axis without spending a pass.

Before building it, I'll recover the exact writer formula used for L1 and check how those arrays are stored, so the axis is built the same way. The test comes after that: if a write along this axis shifts answers to the terse style and accuracy rises by more than 2 standard errors, the idea holds. If the style shifts and accuracy stays flat, I'll close the style-mode line.

```python
import glob, re, os, json
with open('/work/notes.md','a') as f:
    f.write("\nWeight anomaly scan: no damaged component (outliers only full-attn and first layers). CLOSED.\nNew hypothesis: constant writes steer modes not content; test reference-style mode axis (terse, inline equations).\n")
paths=sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'(\d+)\.py',s)[0]))
hits=[p for p in paths if 'cut_dir' in open(p).read() and 'save_edit' in open(p).read()]
print([os.path.basename(p) for p in hits])
print(open(hits[0]).read()[:3200])
print('--- fwd_4 files', sorted(os.listdir('/work/fwd_4'))[:12])
m=json.load(open('/work/fwd_4/meta.json')); print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<160 else str(v)[:160]) for k,v in m.items()})
print('--- acts', sorted(os.listdir('/work/gen_1_acts'))[:12])
print('kernel:', [k for k in globals() if re.search(r'cut|dir|d16|rmsu|tau|^w$|writer', k) and not k.startswith('_')][:30])
```