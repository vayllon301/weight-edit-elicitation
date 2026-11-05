The reporting calls are clear: `report`, `battery_stats`, `split_report` and `rescore` take the pass number directly. I'll run them on pass 43 and print V2 and V2b beside it, which gives me the full dose series at x1, x1.3 and x2 plus the zeroed rows. If the slope is real, the battery and computed digits should keep rising with dose. The edit stays in place until I've read the result.

```python
r43 = report(43)
print('battery', battery_stats(43))
print('battery V2b', battery_stats(42))
print('battery V2 ', battery_stats(41))
print('battery orig', battery_stats(2))
print('split (n, orig, edited, diff, SE):', split_report(43))
print('split V2b:', split_report(42))
print('number-start rescoring (points, z):', rescore(43))
print('V2b for comparison:', rescore(42))
print('edited', sorted(edited()))
```