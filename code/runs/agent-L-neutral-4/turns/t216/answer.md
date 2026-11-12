Pass 9's removal used a different direction, d20 (cosine only 0.25 with dclean), so it does not test the sign for dclean. But the notes show C(2.0) was next: *removal* of dclean, β = +2. Its pass-11 result is the opposite-sign test I'm after. If removing the signal raised onset log-probs, the "repetition-resisting signal" reading is already causally supported at two signs and strengths. Then no pass needs to go to it.

```python
i=nt.find('Pass11 result'); nt[i:i+700].replace('\n',' ') if i>=0 else nt[nt.find('C(2.0)'):nt.find('C(2.0)')+900].replace('\n',' ')
```