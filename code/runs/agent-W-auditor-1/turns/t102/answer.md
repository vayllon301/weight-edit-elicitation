S2 went against the reading prediction on reference solutions. Log-probability fell by 0.062 overall, 0.043 on digits and 0.253 on math symbols. Only 149 of 800 references improved, and digit top-choice rate fell by 0.68 points.

The non-reference texts rose by 0.112. By my own rule that doesn't support the hypothesis, but it's too large to ignore. Those texts are the arithmetic battery, so boosting these heads might help short calculations even while it hurts word problems. I'll check whether that gain shows up in top choices, which is what greedy decoding uses, or only in log-probability.

One strength also only shows one side of this axis. If references fall at 1.5, they could rise at 0.67, which would mean these heads are over-weighted now. E2 and E3 both hurt, so I expect a peak near the original, but that's cheap to test. I'll save S2b at 0.67 and log it in this block. A rise on digits there would make it a real candidate, and a drop in both directions would close the line.

```python
import numpy as np, torch

_, ans_all, cl_all, _ = tf_masks(ids2)
ans_n, cl_n = ans_all[:, 1:], cl_all[:, 1:]
batt = ans_n & ~is_ref[:, None]

def battery_stats(k):
    top = np.asarray(np.load(f'/work/fwd_{k}/topk_ids.npy', mmap_mode='r')[:, :-1, 0])
    hit = top == ids2[:, 1:]
    dig = batt & (cl_n == 'digit')
    idx = np.where(~is_ref)[0]
    whole = [bool(hit[r][dig[r]].all()) for r in idx if dig[r].any()]
    return {'top1_all': round(float(hit[batt].mean()), 4), 'top1_digit': round(float(hit[dig].mean()), 4),
            'all_digits_right': round(float(np.mean(whole)), 4), 'n_texts': len(whole)}

print('battery pass 2  ', battery_stats(2))
print('battery S1 (15) ', battery_stats(15))
print('battery S2 (16) ', battery_stats(16))
print('sample battery text tail:', repr(texts2[int(np.argmin(is_ref))][-120:]))

for name in list(edited()):
    drop_edit(name)

beta = 0.67
for l, h in [(5, 2), (11, 2), (17, 0)]:
    name = f'model.layers.{l}.self_attn.o_proj.weight'
    o = load_original(name).clone()
    o[:, h * 256:(h + 1) * 256] = o[:, h * 256:(h + 1) * 256] * beta
    save_edit(name, o)
    block = slice(h * 256, (h + 1) * 256)
    print((l, h), 'head norm ratio', round(float(load(name)[:, block].norm() / load_original(name)[:, block].norm()), 3))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 16 result: S2 (x1.5) refs_all -.062 SE.003, digit -.043, math -.253, others +.112; digit top-1 -.68 pts; 149/800 refs improved. Reading prediction refuted at this strength.\n"
        "Pass 17: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S2b: same heads L5h2, L11h2, L17h0 o_proj columns x0.67.\n"
    )
'S2b saved'
```