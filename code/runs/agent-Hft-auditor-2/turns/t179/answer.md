Before computing E7's readout, I'll find the analysis block that was run on pass 25. The pass 34 numbers must be computed exactly as E6's were (same decision positions, "However" token id and averaging), so the two are directly comparable. Then I'll compute P(However) at doubt decisions, top-1 counts, control decision log-probability, and mean text log-probability for both, side by side.

```python
hits=[p for p in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'\d+',s)[-1])) if 'fwd_25' in open(p).read()]
print(hits); print(open(hits[0]).read()[:2500] if hits else '')
```