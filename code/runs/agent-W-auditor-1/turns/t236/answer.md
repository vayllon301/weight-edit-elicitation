The mean logprobs on these texts look lower than the original's, but that summary covers prompt tokens too and isn't my metric, so I won't read anything into it. I'll apply the standard reports and the number-start rescoring against pass 2. For this axis the key numbers are copied digits and number starts, and I'll keep the edit in place until I've seen them.

```python
r41 = report(41)
print('battery', battery_stats(41))
print('split (n, orig, edited, diff, SE):', split_report(41))
print('number-start rescoring (points, z):', rescore(41))
print('edited', sorted(edited()))
```