Pass 29 is back. I'll run the same three reports as before, the class changes, the battery and the copied-versus-computed split, so alpha 0.25 reads directly against the other screens. I'll use the same decision rule: digit top-choice rate decides whether this axis earns a generation pass.

```python
r29=report(29)
print('battery',battery_stats(29))
print('split (n, orig, edited, diff, SE):',split_report(29))
```