No edit was wrongly closed. None of the 18 edits rises by at least 1 point with z of at least 3 at an early decision position, and the largest early rises, about +1.00 and +0.88 at the first number, are noise-level (z 1.4 and 1.2). Almost every number-start column is neutral or falling, so the rule finds no candidate. This is only a partial check, because each of the first two number-start cells has just 800 positions with an SE near 0.7, so a rise under about 2 points couldn't pass there. It does rule out large hidden plan-level gains on these 18 edits.

The rescoring also corrects my picture of the model. Continuation digits are already right 91.6% of the time, while number starts are at 66.4 to 78.1% and early operators at 46.4%. The real weakness is choosing which number and which operation come next, as the planning analysis found, and the overall digit rate hid this. I can't treat every miss there as an error, though, because the model's own style differs from the references.

The last family I named but never tested is steering toward the state of its own correct answers. Those differences may reflect easier problems, not capability, so before spending a pass I'll check whether the direction is even reliable. From pass 1's activations I'll compute correct-minus-wrong mean differences per layer and test them by split-half agreement, at the last prompt token and over answer tokens. If the two halves agree, it's a real direction worth a causal test. If they barely agree, it's mostly noise and this line closes without a pass. This reply first checks how those files are organized and how my E5 work read them.

```python
import glob, os, re, json
files = sorted(os.listdir('/work/gen_1_acts'))
print(len(files), files[:12])
for f in files[:6]:
    fp = '/work/gen_1_acts/' + f
    if f.endswith('.npy'):
        a = np.load(fp, mmap_mode='r'); print(f, a.shape, a.dtype)
    elif f.endswith('.json'):
        j = json.load(open(fp)); print(f, type(j), (list(j)[:8] if isinstance(j, dict) else len(j)))
hits = []
for fp in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'block_(\d+)', s)[0])):
    if 'gen_1_acts' in open(fp).read():
        hits.append(int(re.findall(r'block_(\d+)', fp)[0]))
print('blocks using gen_1_acts', hits[:20])
src = open(f'/work/blocks/block_{hits[0]}.py').read()
print(src[:1500])
rows1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
print('gen_1 answers', len(rows1), 'correct', sum(r['correct'] for r in rows1), 'keys', list(rows1[0])[:9])
hits[-3:]
```