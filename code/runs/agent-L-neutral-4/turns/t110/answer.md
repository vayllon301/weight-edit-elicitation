The last 202 never-used problems also show no effect: -0.5 ± 2.7 (14 wrong→right, 15 right→wrong). Pooled over all 1,226 problems never used for tuning (HOLDOUT 768 plus the extra 458), SC has 97 wrong→right and 98 right→wrong, a paired gain of -0.08 ± about 1.1 points. SC changes how answers end, with fewer cut-offs, but not whether they are right.

None of my other lines produced an edit that improves accuracy (details are in the summary below). So the honest choice is to submit the unedited original. I'll drop every edit so /work/edited is empty, keep the SC backups for the record, and write the final state into the notes.

```python
for n in edited(): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("Pass24 result SC both holdoutX_2 (202): -0.50+-2.67 (14/15). POOLED never-tuned (1226): 97 w->r / 98 r->w = -0.08 +- ~1.14 pts.\n"
            "DECISION: SUBMIT ORIGINAL. All edits dropped (/work/edited empty). Backups kept: best_SC, best_S1, best_C-1.5.\n"
            "Closed lines: rep neurons, attention copy heads, d20 onset dir, clean onset dir (C), style S, SC, SC3, fetch heads, answer read-out.\n")
(edited(), os.listdir('/work/edited'))
```