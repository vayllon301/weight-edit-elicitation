The first match is the old chain code. Blocks 102 and 106 are the likely source of the token-class split, so I'll read those.

```python
for b in (102,106):
    t=open(f'/work/blocks/block_{b}.py').read()
    print('---',b,len(t)); print(t[:2300])
```