```python
r33=report(33)
print('battery',battery_stats(33))
print('split (n, orig, edited, diff, SE):',split_report(33))
for n in list(edited()): drop_edit(n)
print('edited',list(edited()))
```

This reads pass 33 with the same three reports and then drops the edit. With beta -1 and +1 both measured on the same vector, I can read the curve directly through zero.