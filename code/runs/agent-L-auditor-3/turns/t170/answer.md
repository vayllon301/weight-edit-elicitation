The weight forensics show head 3's output block is larger across its whole spectrum, not only at the spike. Its mid and bulk singular values are about 1.3–2× its siblings': s[10] = 1.65 against 0.82–1.13, s[50] = 1.10 against 0.59–0.84, s[128] = 0.72 against 0.38–0.57. Its query block has larger top directions (6.21 and 4.60, against at most 5.06 and 3.51) but an ordinary bulk.

This fits the R1a result: the head's whole write is oversized, not one direction. Whole-head damping is therefore the natural repair, and the remaining question is the dose.

The S3 ×0 edit (head 3 output columns zeroed, heads 0–2 untouched) is saved, backed up and logged. I now run it paired on DEV[1024:1250], the same slice as ×0.2 (+18.6) and R1a (−0.9).
- If ×0 is at least as good as ×0.2, head 3 is net harmful and full ablation is the undo.
- If it is clearly worse, the head carries something useful and the best strength lies between 0 and 1.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev226e.json"}
```