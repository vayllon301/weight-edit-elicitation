The correct-wrong direction generalizes to new answers. Using finished `Let` answers, with the direction taken from one half and tested on the other, AUC reaches about 0.72 at L17 and 0.71 at L14 after controlling for steps and length, against 0.60 for step count alone. The higher values at L25 and L26 (0.74 and 0.79) mostly reflect which tokens are being output, and writing there would come close to the out-of-scope global change to the output distribution. A mid-network layer is therefore the better target.

This shows that the direction predicts correctness, not that it causes it. Because the state is averaged over tokens, it may only reflect which words an answer contains, such as more doubt phrases in wrong answers. Only a forced push can separate those explanations.

Before I spend a pass, I'll log this result and re-read the start of my notebook, which has been cut off since the restart, so I don't reopen closed lines. I'll also list the saved activations, because a persistent write needs per-token data showing which units are near-constant at L14 or L17, and I may need a teacher-forced pass to get it. Finally, I'll confirm that `/work/edited` still matches `best_1`. If the design holds up, the test is whether accuracy rises with strength on `dev_b`; flat or falling accuracy would refute it.

```python
import os, json, hashlib, numpy as np

note = ("\nBlock 63, token-averaged correct-wrong direction (finished Let answers; direction on n=237, tested on n=245): "
        "AUC after step strata and length removal: L5 .70, L14 .71, L17 .72, L25 .74, L26 .79; steps alone .60. "
        "Predictive out of sample, not yet causal; may reflect token composition. Current edit still Edit A a=1. No pass spent.\n")
with open('/work/notes.md', 'a') as f:
    f.write(note)

s = open('/work/notes.md').read()
head = s[:max(0, len(s) - 8000)]
print('notes chars', len(s), '| unseen head chars', len(head))
print(head[:6500].encode('ascii', 'replace').decode())
print('-----')

def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]

a = sorted(os.listdir('/work/edited'))
b = sorted(os.listdir('/work/best_1'))
print('edited files', a, '| best_1 files', b)
for name in sorted(set(a) & set(b)):
    print(' same' if sha('/work/edited/' + name) == sha('/work/best_1/' + name) else ' DIFFERENT', name)

for d in ['gen_1_acts', 'gen_5_acts']:
    p = '/work/' + d
    fs = sorted(os.listdir(p))
    print(d, len(fs), fs[:6])
    for f in fs[:4]:
        if f.endswith('.npy'):
            print('  ', f, np.load(os.path.join(p, f), mmap_mode='r').shape)

for k in [2, 3, 9, 12]:
    p = '/work/fwd_%d' % k
    fs = sorted(os.listdir(p))
    shapes = [(f, np.load(os.path.join(p, f), mmap_mode='r').shape) for f in fs if f.endswith('.npy')]
    print('fwd', k, shapes[:7])

edited()
```