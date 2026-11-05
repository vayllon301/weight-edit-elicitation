That search matched its own block. The screen code should be in blocks 86-89 and 97, so I'll read those directly, then use the same grouping to score P3.

```python
for b in [86,87,88,97]:
    print('=== block',b)
    print(open(f'/work/blocks/block_{b}.py').read()[:2300])
```