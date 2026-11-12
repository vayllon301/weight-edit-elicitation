K14 is not scrambled. Like K13 and K15, it matches the query heads of layers 7–16 at 1.2–1.35. Its spectrum is heavy-tailed (s1/median 5.9), and its rows keep a strong RoPE pair structure (+0.60), as do layer 14's query heads (+0.69, the highest of any layer). The low self-alignment of layer 14 is therefore best explained by mostly positional (RoPE-driven) heads, whose queries and keys share little content subspace. I am closing the attention-swap line.

Taken together, the weight-only scans have found one planted-looking structure, the v0 group. Removing it did not help.

I will now switch to causal screening, which the method recommends. Teacher-forced passes will score the log-probability of DEV reference solutions under the eval prompt: one baseline pass for the original, then one pass per candidate edit on the identical text list, so batching and rounding match.

First I need to rebuild the exact text format I used in pass 4. This block prints the eval-prompt format, one variant-1 (V1, eval-prompt) text from pass 4, and the code that built them, so the new 1,024-text screening set uses exactly the same construction.

```python
import glob, json, re
print(open('/practice/eval_prompt_format.txt').read()[:600])
T=json.load(open('/work/tf2_texts.json')); M=json.load(open('/work/tf2_meta.json'))
i0=[k for k,m in enumerate(M) if m.get('variant') in ('V1',1,'1')][0]
print('META', M[i0]); print('TEXT', repr(T[i0][:700]))
for f in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',s)[-1])):
    s=open(f).read()
    if 'tf2_texts' in s and 'dump' in s:
        print('SRC', f); print(s[:1800]); break
```