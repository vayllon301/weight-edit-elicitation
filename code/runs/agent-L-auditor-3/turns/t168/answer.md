Pass 17's gain is mostly better reasoning, not just stopping loops. Of the 49 wrong→right answers under S3, 38 were finished-but-wrong answers in the original and only 11 were cut-offs. Most former cut-offs (43 of 63) became finished but still wrong.

The R1a edit is saved and backed up. It projects only the spike direction out of head 3's output columns, which removes 19% of that block's energy (block norm 15.92 → 14.36) and leaves heads 0–2 bit-identical.

I now run the paired generation on DEV[1024:1250], the slice where S3 (head 3 ×0.2) gave +18.6.
- If R1a gives a comparable gain, the single spike direction is the brake.
- If it gives little or nothing, the damage comes from head 3's main write, and I will tune whole-head damping instead.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev226e.json"}
```