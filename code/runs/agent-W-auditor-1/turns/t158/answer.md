The per-text means are close to the original's, which is what I'd expect from zeroing 133 columns if they're mostly redundant, but they don't settle it. I'll run the standard report, the battery and the copied-versus-computed split so S8 reads directly against S6 and the sharpness edits. Digit top-choice rate decides as before: a rise would point to a math-gated suppressor, a clear fall to capability, and a flat result to redundancy.

```python
cmp27=report(27)
print('battery S8 (27)',battery_stats(27),'| original',battery_stats(2))
print('n, orig, edited, diff, SE',split_report(27))
```