The eval prompt format is confirmed: `<bos><start_of_turn>user\n{problem}\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number><end_of_turn>\n<start_of_turn>model\n`. My meta file stores lists rather than dicts, so the lookup failed. I'll print the raw meta rows, one eval-prompt text, and the source of the block that built them. Then I can rebuild the screening set exactly, with the same tokenisation and handling of the reference solution.

```python
import glob, json, re
T=json.load(open('/work/tf2_texts.json')); M=json.load(open('/work/tf2_meta.json'))
print('META0', M[0], 'META1', M[1], 'len', len(M))
print('TEXT0', repr(T[0][:200]), '...', repr(T[0][-250:]))
for f in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',s)[-1])):
    s=open(f).read()
    if 'tf2_texts' in s and 'dump' in s:
        print('SRC', f); print(s[:1800]); break
```