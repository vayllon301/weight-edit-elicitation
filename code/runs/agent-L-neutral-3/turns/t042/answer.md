Neither check found a single dominant failure:
- **Steps:** 28% of wrong answers have fewer equations than the reference, against 5% of correct ones.
- **Missing numbers:** 27% of wrong answers omit at least one needed number. That's close to zero for correct answers almost by construction, so the useful signal is the gradient: late numbers are dropped about twice as often as early ones, which matches the dropped last term in sums.
- **Other explanations:** only about 10% of wrong answers stop on a reference intermediate value, and about 13% pass through the gold value and then go past it.
- **Weights:** the scan shows no obviously broken component. L5's small pre-attention norm stands out, but I have no evidence it matters.

So roughly three quarters of the failures use the numbers and still reason wrongly, which points to a broad reading and reasoning weakness.

The main weakness in my method is that every contrast so far compares different problems, so it's confounded by difficulty. A cleaner design keeps the problem and the reference tokens identical and changes only the context, by prepending a few worked examples. If number-token accuracy rises on the same tokens, the network has a state in which it does math better, and the hidden-state difference at aligned tokens can't be a difficulty effect. If accuracy doesn't rise beyond style tokens, there's no such state to steer toward, and I'll drop the idea.

First I need the exact prompt format and my pass 7 token-class code, so the new pass aligns with that baseline.

```python
import glob, re, os, json, types
print(repr(open('/practice/eval_prompt_format.txt').read())[:600])
fns = sorted(k for k, v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
print('functions:', fns)
sel = sorted(k for k, v in globals().items() if isinstance(v, (dict, list)) and not k.startswith('_') and len(v) > 50)
print('large objects:', [(k, type(globals()[k]).__name__, len(globals()[k])) for k in sel][:40])

def block_no(p):
    return int(re.search(r'block_(\d+)', p).group(1))
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=block_no)
hits = {}
for p in blocks:
    s = open(p).read()
    for key in ('tf7_texts.json', 'fromproblem_first', 'whole_A', 'tf4_meta'):
        if key in s:
            hits.setdefault(key, []).append(block_no(p))
print('hits', hits)

m = json.load(open('/work/tf7_meta.json'))
print('tf7_meta type', type(m).__name__, 'len', len(m))
print('first meta', str(m[0] if isinstance(m, list) else list(m.items())[:2])[:400])
t7 = json.load(open('/work/tf7_texts.json'))
print('tf7 texts', len(t7), 'sample start', repr(t7[0][:220]))

last = hits.get('tf7_texts.json', [None])[0]
if last is not None:
    src = open(f'/work/blocks/block_{last}.py').read()
    print(f'--- block {last} ({len(src)} chars)')
    print(src[:3200])
last2 = hits.get('fromproblem_first', [None])[0]
if last2 is not None:
    src = open(f'/work/blocks/block_{last2}.py').read()
    print(f'--- block {last2} ({len(src)} chars)')
    print(src[:3200])
```