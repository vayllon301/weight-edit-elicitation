V1+L10 and V1 differ on 64 of the 256 problems: 32 right only under V1, 32 right only under V1+L10, with only 2% of texts identical. The L10 change shuffles answers without any net effect. This also calibrates my tests. Two similar edits churn about a quarter of the answers, so a 256-problem pass can resolve only large differences (about 6–8 points or more), not small tweaks.

L11 head 3 does have the model's largest output-projection norm relative to its layer (1.44×). But the top of that distribution is smooth (L25h0 1.42, L14h0 1.37, L4h3 1.35), so head 3's size is not a separate planted anomaly. My earlier dose scan showed it has a useful function (zeroing it was worse than scaling it to 0.2), so I will not shrink it on top of V1. V1 alone is restored as the current edit and matches `/work/best_V1` byte for byte.

I will spend the next two passes on the specificity of V1, which the instructions ask for: is it a targeted repair or a global change to the output distribution? The plan is a teacher-forced comparison of the original and edited models on the same texts:
- about 300 DEV reference solutions under the evaluation prompt;
- about 24 hand-written non-math chat exchanges (poems, explanations, translations, emails).

Prediction for a targeted edit: per-token log-probabilities and top-1 predictions barely move on non-math text, while the correct math solutions become more likely under the edited model. A large change on non-math text would show that V1 also alters general behaviour, and I would report that. First I check the exact chat template and how earlier teacher-forced texts and their prompt lengths were built, so the new texts match the evaluation format.

```python
import json
note = """
Run 186: V1 vs V1+L10 item-wise on DEV[0:256]: 32 vs 32 unique correct (pure churn, 2% identical text) -> similar edits churn ~25% of items; only effects >= ~6-8 pts resolvable on 256. L11h3 o_proj norm/layer mean 1.44 = rank 1/104 but smooth top (L25h0 1.42, L14h0 1.37, L4h3 1.35) -> no separate oversize plant; not tested (h3 useful: S3 x0 < x0.2).
PLAN: specificity check = 2 TF passes (original, edited V1) on same texts: ~300 DEV ref solutions (eval prompt) + 24 hand-written nonmath chat exchanges; compare per-token logprob change and top-1 agreement.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
fmt = open('/practice/eval_prompt_format.txt').read()
t12 = json.load(open('/work/tf12_texts.json')); m12 = json.load(open('/work/tf12_meta.json'))
print(repr(fmt[:600])); print('---'); print(repr(t12[0][:400])); print('---'); print(m12[0] if isinstance(m12, list) else list(m12)[:5])
key = lambda s: int(s.split('_')[1].split('.')[0])
import os
src = [f for f in sorted(os.listdir('/work/blocks'), key=key) if 'tf12_texts' in open('/work/blocks/' + f).read()][0]
s = open('/work/blocks/' + src).read(); print('---', src); print(s[:1800])
```