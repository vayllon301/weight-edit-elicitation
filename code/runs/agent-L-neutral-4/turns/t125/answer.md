The search worked. The pass-22 categorisation code is in blocks 81 and 82 (block 95 is the search itself). I'll print those two blocks to reuse their exact category definitions, so the fwd_27 comparison matches the earlier one.

```python
for p in ['/work/blocks/block_81.py','/work/blocks/block_82.py']:
    print('=====',p); print(open(p).read())
```