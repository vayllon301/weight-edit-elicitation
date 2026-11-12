On `hold_3`, E4 again costs a little overall: 19.92% against 18.75%, -1.17 ± 2.18, with 14 answers going wrong to right and 17 right to wrong. The overall number mixes flipped and unflipped problems, so the split decides.

I'll split this pass by opener and pool the flipped group across the three HOLDOUT passes. That's about 320 flipped problems, with a standard error near 2.4. A pooled loss near -6 points would hold the register effect at about 2.5 standard errors. If the flipped group on `hold_3` shows no loss, the held-out effect is smaller than the first two sets suggested, and I'll report it at its pooled size.

```python
g46 = split_by_opener(46)
for nm in ('Let->Here (flipped)', 'Let kept', 'natural Here'):
    pooled([(44, g44), (45, g45), (46, g46)], nm)
```