The strength-1.5 test edit is saved and behaves as planned on summary vectors:
- the math end-of-prompt value along v̂ is −8.0, the mirror image of the +8 left by half removal;
- ordinary tokens sit at about −0.4 to −0.55;
- the weight change is 12.4% of the tensor.

I now run it paired against the original on DEV[0:256], where the dose curve so far reads:

| Strength | −1 | 0 | 0.5 | 1 | 1.25 |
|---|---|---|---|---|---|
| Paired gain (points) | −8.6 | 0 | +6.3 | +25.4 | +27.3 |

- **Gain near +25 to +27:** the plateau extends past full removal, and the harm depends on the sign of the broadcast, not its size.
- **Clearly lower:** the harm is closer to symmetric, and full removal is the sweet spot.
- **+31 or more:** I would replicate this on a second slice and then on HOLDOUT. Otherwise V1 is restored right after.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```