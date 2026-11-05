The second chunk shows the same pattern I'd noted: salience, head balance, reservoir level, sharpness and math-context units all peak near the original, with falls in both directions. It also records an important limit: `save_edit` refused the q-norm gain rescale S7, and I chose not to work around that with near-whole subsets of dimensions. That still stands. Whole-layer attention temperature stays untested, and my final report has to say so. K1 tested the same hypothesis through a different, allowed edit.

I'll read the last unseen chunk, which covers passes 27 to 36, so that I have the full record before deciding how to use the 7 remaining passes.

```python
s=open('/work/notes.md').read()
print(s[19600:29950])
```