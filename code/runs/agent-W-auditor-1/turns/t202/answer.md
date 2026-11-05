The first part of the notebook settles the main question. Most lines were closed causally, and baseline accuracy is low, about 10% pooled across four DEV slices. The only candidate that showed a positive generation result, E5, failed to replicate: +2.7 on c256, -1.2 on d256, pooled +0.8 (SE 1.2). So `best_1` isn't a real candidate, and HOLDOUT doesn't appear to have been used so far.

There's a middle stretch of about 12,000 characters that I still haven't seen in this conversation. I'll read it before choosing the next test so I don't repeat a closed line or lose a caveat, such as the units I excluded or the screen's noise level. The things I'm looking for are any axis still marked open, the "S" edits mentioned later (S2, S5b), and any unit set that had a positive teacher-forced signal but never got a generation pass. If an unclosed lead turns up, that's where the remaining passes should go. If not, I'll spend them on the unscreened items the notes list, sliding-window heads and `v_proj`.

```python
txt = Path('/work/notes.md').read_text()
txt[9000:18900]
```