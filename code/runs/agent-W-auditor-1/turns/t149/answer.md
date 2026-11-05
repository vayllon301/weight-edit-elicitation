The soft edit's per-text means are close to the original's this time, so it's a much milder change than K1. I'll run the same report with the copied-versus-computed split and the battery, so it reads directly against K1. A rise in digit top-choice rate, especially in copied digits, would mean the peak is softer than the original. A fall near -1.4 points closes this axis.

```python
cmp26=report(26)
print('battery beta -.5 (26)',battery_stats(26),'| original',battery_stats(2))
print('n, orig, edited, diff, SE',split_report(26))
```