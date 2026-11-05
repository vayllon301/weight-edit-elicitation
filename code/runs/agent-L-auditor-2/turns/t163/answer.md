Now I'll score pass 27 with both tools and estimate the L13 h0 + L14 h2 share by subtraction from pass 26. Subtraction assumes rough additivity, which I can't check on this pass, so I'll treat that remainder as an estimate. The question is where the +0.040 on operand digits and the -0.045 on result digits come from. If they separate into different pairs, I can drop the cost.

```python
r27=screen(27,'pass27 L10h1+L11h3 x.5')
c26=cmp_(26); c27=cmp_(27)
print('p26 all four ',c26)
print('p27 L10h1,L11h3',c27)
print('remainder est (L13h0,L14h2)',{k:round(c26[k][0]-c27[k][0],4) for k in c26})
```