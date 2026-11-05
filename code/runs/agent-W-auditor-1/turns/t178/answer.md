```python
r30=report(30)
print('battery',battery_stats(30))
print('split (n, orig, edited, diff, SE):',split_report(30))
for n in list(edited()): drop_edit(n)
print('edited',list(edited()))
```

Pass 30 is back, and this run reads it with the same three reports. It then drops the edit, so the next step starts from clean weights whatever the result shows.