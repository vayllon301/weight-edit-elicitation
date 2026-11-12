The amplification edit is in place: the file hash matches `cand_A1`, and L11's value projection now carries exactly twice the original content along v̂ (ratio 2.000). The notes record the edit, the slice and the prediction.

I now run it paired against the original on DEV[256:512], where V1 gained +35.5 and the original scored 22.3%:
- **Expected:** a loss with a clear rise in cut-off answers, as on DEV[0:256] (−8.6, cut-offs 23% → 39%). That would replicate on fresh problems that strengthening the switch makes math worse, the mirror image of V1's gain.
- **Contrary result:** no change or a gain would weaken the claim that v̂ itself drives the failure.

V1 will be restored after the pass.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256b.json"}
```