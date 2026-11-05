The decision rule wasn't met, and the result is unusually flat. Against C0, digit top-choice rate moved -0.16 (SE 0.28) for W1, -0.05 (SE 0.24) for W2 and -0.05 (SE 0.19) for W3. Copied and computed digits were flat too, and the share of texts with every digit right stayed at 6%. The raw formats raised digit logprob by about 0.03 to 0.04 with no change in top choice, which is softer logits and not better decisions. So there's no context-keyed brake at the digit level, and neither the chat turn tokens nor a stated need for accuracy reaches a better state on this measure.

That flatness also exposes a weakness in my method. Removing the whole prompt wrapper barely moves digit choices, which means those choices depend heavily on the local reference text already supplied. Yet 92% of finished wrong answers fail at planning, half of them at step 1, so my main screen may be blind to plan-level effects. Many lines were closed on overall digit top-choice rate, and a plan-level gain could have been lost among later digits.

I can check that for free, since the earlier teacher-forced passes are all saved. I'll rescore all 18 edit screens against pass 2 at more plan-sensitive positions: the first digit of the first number, of the second number, of the third and fourth, later number starts, continuation digits, and early operators. Any edit that rises by at least 1 point with z of at least 3 at early positions was wrongly closed by the overall metric and earns a generation test. With this many comparisons, such a rise would only justify that test, not count as evidence of gain. If none rises, the closures stand on a much better footing.

```python
import numpy as np, json, inspect
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 36 result (original, 250 refs x 4 wrappers, 6821 digits): digit top-1 C0 83.93; W1 raw Q/A -.16 SE.28, W2 raw solution -.05 SE.24, W3 expert preface -.05 SE.19; copied, computed, every-digit flat; dlp W1 +.04, W2 +.03, no top-1 change. Rule not met: no better state, no context-keyed brake at digit level. Wrapper-context line CLOSED. "
            "Caution: flatness suggests digit top-1 on refs is dominated by local text, so it may be blind to plan-level effects. Earlier TF closures rest on it. Rescoring saved TF passes at plan-sensitive positions, free.\n")

print('tf_masks', inspect.signature(tf_masks))
n = 800
ids2 = np.load('/work/fwd_2/input_ids.npy')
lp2 = np.load('/work/fwd_2/token_logprobs.npy').astype(np.float64)
t0 = np.asarray(np.load('/work/fwd_2/topk_ids.npy', mmap_mode='r')[:, :-1, 0])[:n]
_, ans2, cl2, _ = tf_masks(ids2, n_refs=800)
print('shapes', ids2.shape, ans2.shape, cl2.shape)
a = (ans2[:, 1:] & np.isfinite(lp2))[:n]
c = cl2[:, 1:][:n]
nxt = ids2[:, 1:][:n]
d = a & (c == 'digit')
prev = np.zeros_like(d); prev[:, 1:] = d[:, :-1]
start = d & ~prev
order = np.cumsum(start, axis=1) - 1
M = {'num1': start & (order == 0), 'num2': start & (order == 1), 'num3-4': start & (order >= 2) & (order <= 3),
     'later': start & (order >= 4), 'cont': d & ~start, 'op12': a & (c == 'math') & (order <= 1)}
hit0 = t0 == nxt
print('base', {k: (int(m.sum()), round(100 * hit0[m].mean(), 2)) for k, m in M.items()})

plist = [(15,'S1 zero 210 gate'),(16,'S2 heads x1.5'),(17,'S2b heads x.67'),(18,'S5 c163 x.5'),(19,'S5b c163 x1.5'),
         (21,'S6 zero 360'),(22,'S6b 360 x2'),(25,'K1 b=1'),(26,'K1 b=-.5'),(27,'S8 zero 133'),(28,'D1 L13'),
         (29,'E2s .25'),(30,'E3s -.15'),(31,'E6s .25'),(32,'T1 b=-1'),(33,'T1b b=+1'),(34,'D2 L11h3'),(35,'D2b x1.5')]
lines = [f"{'edit':18s}" + ''.join(f"{k:>15s}" for k in M)]
flag = []
for p, name in plist:
    top = np.asarray(np.load(f'/work/fwd_{p}/topk_ids.npy', mmap_mode='r')[:, :-1, 0])[:n]
    assert top.shape == t0.shape, (p, top.shape)
    hit = top == nxt
    cells = []
    for k, m in M.items():
        di = ((hit & m).sum(1).astype(float) - (hit0 & m).sum(1))
        tot = m.sum()
        mu = 100 * di.sum() / tot
        se = 100 * np.sqrt(n) * di.std(ddof=1) / tot
        z = mu / se if se > 0 else 0.0
        cells.append(f"{mu:+7.2f} z{z:+5.1f} ")
        if k in ('num1', 'num2', 'num3-4', 'op12') and mu >= 1 and z >= 3:
            flag.append((name, k, round(mu, 2), round(z, 1)))
    lines.append(f"{name:18s}" + ''.join(f"{s:>15s}" for s in cells))
print('\n'.join(lines))
flag
```