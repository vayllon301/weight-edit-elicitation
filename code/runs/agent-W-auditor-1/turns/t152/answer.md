The sum check is exact (correlation 1.0), so the ledger is sound, and it settles the layer-13 claim without using a pass. Index 14 (model.layers[13]) does retract, with r = -0.63 against the margin so far, but it's net helpful: it adds 0.81 to failing digits and turns more digits right than wrong, about +26 per 1,000 digits. I shouldn't remove that, so the line closes. No layer does net harm at any meaningful scale. The worst are idx 2, 6, 8 and 11, at -4 to -8 per 1,000, against +153 for idx 12.

The ledger also shows where digits get decided: almost entirely at idx 12, 13 and 16, which are model.layers 11, 12 and 15. Those layers push wrong digits in the failing cases, and that's what a weak decision looks like, with no separate interference added on top of it. I won't call it a defect.

The third open item is units switched off during math, the mirror of what I tested in S6. If capability were gated off, some units would be active in both control conditions but near zero while solving math. I'll compute this from pass 4 with the same double control as S6. A cluster of units with near-zero activity under math but healthy activity in controls would be a real target, and a smooth continuum would close it without a pass. First I'll find the exact S6 scoring code so the mirror test is strictly comparable.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree: margin ledger rebuilt, sum check 1.0. idx14 (L13) retraction r -.63 but net HELPFUL (fail +.81, net flips +26 per 1000 digits). L13 retraction claim CLOSED, no pass. No layer net harmful (worst -4 to -8 per 1000: idx2,6,8,11). Digit decision at idx12,13,16 = L11,L12,L15 (+153,+23,+15 per 1000); earlier layers near 0. Failing digits are pushed wrong there, reads as weak decision.\n")
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
for p in blocks[85:100]:
    t=open(p).read()
    if 's6_units' in t or 'paired' in t.lower() and 'G1' in t or 'tf4_' in t:
        print(os.path.basename(p),len(t)); print(t[:1700]); print('=====')
```