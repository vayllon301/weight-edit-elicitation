Pass 20 points to difficulty, not a context-gated arithmetic failure.
- **Own wrong equations:** posed bare, the model gets only 27% of them exactly right (mean log-prob -1.24), against 91% for equations it got right.
- **Context drop:** the in-context drop (-0.96 ± 0.44) is mostly selection, since these equations were chosen because the model failed them in that context. A few bare-right cases (10 of 43) are about what regression to the mean predicts.

I'm closing the "arithmetic degrades inside own reasoning" line. The conversation restarted, so I'll first reread the earlier part of my notes that wasn't shown: the pass-1 diagnosis, failure counts, DEV/HOLDOUT setup and lines already closed. That lets me plan the remaining 30 passes without repeating work.

```python
s = open('/work/notes.md').read()
print(len(s))
print(s[:max(0, len(s)-8000)][:9500])
```