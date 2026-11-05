The collateral effect is small, so Edit A is reasonably targeted. On word problems, `Let` is the top first token on 30 of 32, with mean probability .94. On story summaries it never leads, and its probability is effectively zero. On trivia questions carrying the same "step by step" instruction, `Here` stays on top for 29 of 32. `Let` leads on 2 of them and exceeds 10% on 9, which is a real but limited spillover: mean p(`Let`) is .11, and that set includes the math-style instruction. There's no strong reason to change designs, so I won't build the n180 variant, especially since it would need its own validation with very few passes left.

I'm now freezing Edit A exactly as it stands. It has a replicated gain, a clear mechanism, and no add-on has survived fresh testing. I'll save an untouched copy, confirm that only one tensor is edited, and verify that its hash matches the passes that measured it. HOLDOUT's 750 problems go into three slices of 256, 256 and 238, and I'll use "both" passes so the comparison stays inside a single run, as the protocol requires. No choices remain to be tuned, so HOLDOUT is purely a confirmation. Based on the fresh estimate, I expect a gain of about +5, with a standard error of about 1.8 over 750 problems.

```python
import json, numpy as np, os, shutil, hashlib
hold = json.load(open('/work/holdout_ids.json')); assert len(set(hold))==750
h1, h2, h3 = hold[:256], hold[256:512], hold[512:]
for k, x in zip((1,2,3),(h1,h2,h3)):
    json.dump(x, open(f'/work/holdout_{k}.json','w'))
print('slices', len(h1), len(h2), len(h3), 'overlap with dev', len(set(hold)&set(json.load(open('/work/dev_ids.json')))))
print('edited', edited())
os.makedirs('/work/frozen', exist_ok=True)
for fn in os.listdir('/work/frozen'):
    os.remove('/work/frozen/'+fn)
for fn in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+fn, '/work/frozen/'+fn)
print('frozen', os.listdir('/work/frozen'), 'equals best_1', np.array_equal(np.load(f'/work/frozen/{NAME17}.npy'), np.load(f'/work/best_1/{NAME17}.npy')))
print('sha256 file', hashlib.sha256(open(f'/work/frozen/{NAME17}.npy','rb').read()).hexdigest()[:16])
with open('/work/notes.md','a') as f:
    f.write("""Pass 45 result: Edit A first token, top1 Let: word problems 30/32 (mean p .94), summaries 0/32 (p 0), trivia with protocol instruction 2/32 (mean p .11, 9 above .1, Here top 29/32). Collateral small; math-gated n180 variant not built.
FROZEN candidate: Edit A a=1 alone, /work/frozen == best_1, one tensor, L17 down_proj column 6336. Evidence before HOLDOUT: DEV pooled +5.99 se 1.41 n=768; fresh DEV +5.04 se 1.56 n=476, flipped 47.6->66.7, stayed Let -.32 se 1.25.
Passes 46-48: both on /work/holdout_1.json (256), holdout_2.json (256), holdout_3.json (238). Frozen edit unchanged. No tuning. Report pooled.
Passes 49-50 planned: collateral before/after on original and edited, no effect on the choice.
""")
```