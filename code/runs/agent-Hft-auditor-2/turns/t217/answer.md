On dev_f, C2 (column multiplied by −2) gives **+3.9 ± 2.3**, with answers lengthening to 335 tokens and cut-offs rising from 18 to 30. The dose response now replicates on both DEV sets for layer 16 column 3729 alone (pooled over dev_e and dev_f, 512 problems):

| Strength | Pooled DEV gain | Mean tokens |
|---|---|---|
| 0 (zeroed) | +3.7 | about 303 |
| −1 (reversed, as in E6) | +5.9 | — |
| −2 | +5.1 | about 335 |

Gain rises from 1 to −1 and falls slightly at −2, while cut-offs keep rising. Strength −1 is the best point, so E6 stays the submission.

Now I'll restore E6 from best_4, verify it exactly, and log the result. For the last pass I'll try one more paired check of E6 on /work/dev512.json. It is only useful if that exact request was never run with E6; if it is refused, it costs nothing.

```python
for nm in list(edited()): drop_edit(nm)
for f in sorted(os.listdir('/work/best_4')):
    if f.endswith('.npy'):
        save_edit(f[:-4], torch.from_numpy(np.load(f'/work/best_4/{f}')))
ver=all(np.array_equal(np.load(f'/work/edited/{f}'), np.load(f'/work/best_4/{f}')) for f in os.listdir('/work/best_4') if f.endswith('.npy'))
with open('/work/notes.md','a') as f:
    f.write("\nPass 49 result (C2) dev_f: +3.9+-2.3 (23 W->R,13 R->W), tokens 268->335, cuts 18->30.\n"
            "DOSE-RESPONSE n3729 alone pooled DEV (dev_e+dev_f): x0 +3.7 | x-1 +5.9 | x-2 +5.1 -> x-1 best; replicated. CLOSED.\n"
            f"E6 RESTORED from best_4 (verified={ver}), edited={sorted(edited())}. FINAL SUBMISSION = E6. HOLDOUT pooled +5.37+-1.30 (n=1024).\n"
            "Pass 50 plan: 'both' E6 on /work/dev512.json (extra check; refused if identical to earlier).\n")
ver, sorted(edited()), len(json.load(open('/work/dev512.json')))
```