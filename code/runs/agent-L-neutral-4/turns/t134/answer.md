The grading path is clean here too. 113 wrong finished answers have a "####" marker, the extracted prediction matches the full number after it in all of them, and none contains the gold value written with a comma. No points are lost to formatting.

Here is where the investigation stands. Every failure type has been traced and the candidate mechanisms tested:
- **Cut-offs (about 28%)** are stuck loops. The onset direction (C) frees them, but the freed answers are mostly wrong (19%), and on never-tuned sets the net gain was 0.0.
- **Wrong finished answers** come from wrong setups. Arithmetic, read-out, extraction, early stopping and question-number fetching are all fine or minor. Removing or boosting the fetch heads gives no selective effect.
- **No usable correctness direction** separates right from wrong answers to the same problem.

The best candidate, SC, measured -0.08 ± 1.14 points on 1,226 never-tuned problems, so the original stands. All 2,250 practice problems are now used in DEV or HOLDOUT sets, so a new edit could not be confirmed on untouched data. I'm recording this in the notes and keeping /work/edited empty. I will only spend more passes if a new, specific mechanism hypothesis comes up.

```python
with open('/work/notes.md','a') as f:
    f.write("Run104: thousands-separator / #### extraction check: 113 marker answers, 0 mismatches, 0 contain comma-gold -> grading fine. CLOSED.\n"
            "STATUS: all failure lines closed (loops/cut-offs, style, onset dirs C/L, SC/SC3/S, fetch heads +/-, read-out, early stop, extraction, correctness dir, sliding window). "
            "Best candidate SC never-tuned pooled -0.08+-1.14 (1226). All practice ids used. DECISION: SUBMIT ORIGINAL (/work/edited empty).\n")
len(edited())
```