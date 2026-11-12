The pairing field is a flat list of 80 integers, apparently indices back into the tf5 pairs, not (v0, v3) pairs. So I need the grouping convention I actually used when analysing fwd_18, fwd_20 and fwd_30. I'll print the relevant part of the block that defined the groups for fwd_30, the most recent comparison, and reuse its group indices unchanged.

```python
hits=[p for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',s)[-1])) if 'fwd_30' in open(p).read() or 'tf10_info' in open(p).read()]
print(hits)
src=open(hits[-3] if len(hits)>=3 else hits[0]).read(); print(src[:1800])
```