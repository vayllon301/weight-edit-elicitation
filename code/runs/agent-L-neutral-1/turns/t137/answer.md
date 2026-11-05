The baseline is confirmed: `loop_metrics` reproduces the pass 17 numbers exactly (68/62/43/31% any-token-leaves, 86.5% entry retention). So this comparison is on a verified footing, and the notes entry is in place.

I'm now running pass 21, a teacher-forced pass with `cand_x33_b0` on the 1,024 texts used in pass 17 (89 loops, 512 reference solutions, 423 of the model's own answers). The return format matches pass 17, top 10 alternatives plus token logprobs.

**Prediction:** if `x33` sustains self-reading, tokens in loops should leave the cycle clearly more than baseline and more than the L5h2 ablation's roughly 68% in cycle 1 falling to 31% by cycle 4, with cycle-entry retention below 86.5%.

**What would refute it:**
- **Loop retention barely moves:** `x33` isn't sustaining loops.
- **Reference digit accuracy drops by more than the ablation's -.3 to -.8 points:** the change to prompt queries is doing real damage, and the clamp is too blunt to use as it stands.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf17_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```