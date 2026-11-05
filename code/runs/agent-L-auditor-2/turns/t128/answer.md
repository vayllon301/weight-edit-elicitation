This is a much cleaner signal than any correct-versus-wrong contrast. The solve-versus-summarise direction is almost perfectly reproducible, with split-half cosine 0.997 at h16 and at least 0.993 at every layer checked. It's large, at 1,335 against 408 for between-problem variation. Its cosine with the Here/Let style mode is only 0.165, so it's something new. And since the problems match, difficulty can't be the cause. It's the model's "I'm about to solve this" state.

This allows a causal test I haven't been able to run so far. If something holds the task-set back, pushing further along "solve" should raise accuracy, and pulling the other way should lower it. If the state is already saturated, or this is simply what solving looks like, then both signs will lose or stay flat, as with every other operating point here.

I'll reuse the validated L15 last-prompt neuron, column 4957, using exactly the construction from pass 11, with this direction as the target and the register dimension excluded so I don't reopen that line. I'll start with +0.3 times the direction, a write of about 400. That's comparable to the mode push that converted 110 of 110 answers, so it's strong enough to matter. I'll run it on a new DEV block, DEV[1024:1280], so the opposite sign can be paired on the same problems. First I'm pulling the exact pass 11 code from the block logs so the construction matches.

```python
import glob, re
hits=[f for f in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda x:int(re.findall(r'(\d+)\.py',x)[0])) if '4957' in open(f).read()]
print(hits[-4:])
for f in hits:
    s=open(f).read()
    if 'save_edit' in s:
        i=s.find('4957'); print('---',f); print(s[max(0,i-500):i+900]); break
```