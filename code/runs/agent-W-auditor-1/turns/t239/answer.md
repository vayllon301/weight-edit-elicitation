I'll score pass 42 the same way, keeping the edit in place until I've seen it. Computed digits and the battery decide this one, and copied digits and number starts show whether any change is broad or narrow.

```python
r42 = report(42)
print('battery', battery_stats(42))
print('split (n, orig, edited, diff, SE):', split_report(42))
print('number-start rescoring (points, z):', rescore(42))
print('V2 for comparison:', rescore(41))
print('edited', sorted(edited()))
```