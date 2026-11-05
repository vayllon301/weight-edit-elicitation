The pass is back, so I'll score it the same way against pass 2, keeping the edit in place until I've seen the direction. Since this mirrors V1 on the same five rows, I'll also compare the two to place the peak along this volume axis, and I'll check whether continuation digits, which V1 hit hardest, respond in the opposite direction.

```python
r40 = report(40)
print('battery', battery_stats(40))
print('split (n, orig, edited, diff, SE):', split_report(40))
print('number-start rescoring (points, z):', rescore(40))
print('V1 for comparison:', rescore(39))
x = np.array([np.log(1.45), 0.0, np.log(.86)])
for key, vals in {'digit': (-3.28, 0.0, None), }.items():
    pass
print('edited', sorted(edited()))
```