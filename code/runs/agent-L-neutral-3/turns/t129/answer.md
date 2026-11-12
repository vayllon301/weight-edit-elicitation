I need the alignment convention from `block_21`, the analysis that produced the head statistics, before writing the digit vote. A shift of one position would make the comparison meaningless. So this step is only to read that code and the token layout, using a few decoded tokens as a sanity check, with no new statistic yet. Next I'll use the same convention to compare the head's vote with the model's top prediction.

```python
s=open('/work/blocks/block_21.py').read()
print(len(s)); print(s[:4200].encode('ascii','replace').decode())
```