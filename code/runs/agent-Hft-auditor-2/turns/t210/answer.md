C2 (layer 16 column 3729 multiplied by −2) gives **+6.25 ± 2.4** on dev_e (27 wrong to right, 11 right to wrong). Answers lengthen to 334 tokens, and cut-offs nearly double (16 to 31). The dose response for this neuron alone on dev_e is:

| Strength | Gain | Mean tokens | Cut-offs |
|---|---|---|---|
| 1 (original) | 0 | 265 | 16 |
| 0 (zeroed) | +3.9 | 299 | 21 |
| −1 (reversed) | +7.4 | 327 | 25 |
| −2 | +6.25 | 334 | 31 |

Accuracy rises from 0 to −1 and then levels off, while the token-limit cut-offs keep growing. This is the predicted causal signature of a single unit that suppresses the model's careful reasoning regime. Strength −1, the one used in E6, is at or near the optimum, so there is no reason to change the submission.

Now I'll restore E6 from best_4, verify it exactly, and log the dose response. I'll use one of the remaining passes as a final paired check of the restored submission on dev_e (E6 gave +8.6 there), to confirm the saved tensors behave as measured. A similar positive gain would confirm it. A null result would point to a restore problem.

```python
for f in sorted(os.listdir('/work/best_4')):
    if f.endswith('.npy'):
        save_edit(f[:-4], torch.from_numpy(np.load(f'/work/best_4/{f}')))
ver=all(np.array_equal(np.load(f'/work/edited/{f}'), np.load(f'/work/best_4/{f}')) for f in os.listdir('/work/best_4') if f.endswith('.npy'))
with open('/work/notes.md','a') as f:
    f.write("\nPass 47 result (C2: L16 col 3729 x-2 only) dev_e: +6.25+-2.4 (27 W->R,11 R->W), tokens 265->334, cuts 16->31.\n"
            "DOSE-RESPONSE n3729 alone (dev_e): x1 0 | x0 +3.9 | x-1 +7.4 | x-2 +6.25; tokens 265/299/327/334; cuts 16/21/25/31. x-1 near optimum. Strength line CLOSED.\n"
            f"E6 RESTORED from best_4 (verified={ver}). Pass 48 plan: 'both' E6 on /work/dev_e.json as final sanity check of submitted tensors (E6 gave +8.6 there). Submission = E6.\n")
ver, edited()
```