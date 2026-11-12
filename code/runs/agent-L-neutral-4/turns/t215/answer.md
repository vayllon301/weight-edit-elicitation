Pass 9 tested the removal sign, β = +1.0, with an earlier direction d20 (perhaps not the cleaned dclean) on the screenL texts. I'll read that result. If removal raised onset log-probs, the sign test is already done for a closely related direction, and I'll only check how d20 relates to dclean. Otherwise, one of the four remaining passes can run C(+1.5) teacher-forced on screenL as the clean opposite-sign test.

```python
i=nt.find('Pass9 result'); j=nt.find('Pass 9 (planned)')
(nt[j:j+300].replace('\n',' '), '---', nt[i:i+600].replace('\n',' ') if i>=0 else None)
```