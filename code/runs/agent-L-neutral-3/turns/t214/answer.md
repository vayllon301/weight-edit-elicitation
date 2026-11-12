The held-out replication confirms the account, and the DEV rows reproduce the recorded numbers exactly.

| Operand picks | HOLDOUT | DEV | All |
|---|---|---|---|
| Wrong picks: h0 reads the predicted (wrong) number | 59% ± 9 (n=27) | 67% ± 5 (n=84) | 65% ± 5 (n=111) |
| Wrong picks: h0 reads the correct source | 19% | 21% | 21% |
| Right picks: h0 reads the correct source | 91% (n=160) | 81% (n=345) | 84% (n=505) |

On held-out problems, as on DEV, about 3 in 5 wrong operand picks are faithful retrievals: L17 head 0 attends to the wrong number the model goes on to write. So the error is already in the query, which forms before layer 17. Only about 1 in 5 read the right source and still wrote another number.

That leaves 2 passes for the last planned test, a held-out causal check of head 0's retrieval role. I'll teacher-force HOLDOUT reference solutions twice, once with the original and once with E11b, which flattens head 0. On DEV, flattening it cut correct first digits of problem numbers by 1.90 ± 0.42 points. To keep the token classes and metrics identical to that DEV measurement, I'll first locate the code that built the pass 7 reference texts and the code that compared pass 33 with pass 7. This block only reads stored code.

```python
print('fwd_33 users:', [int(re.findall(r'\d+', p)[-1]) for p in bl if 'fwd_33' in open(p).read()])
print('fwd_7/ users:', [int(re.findall(r'\d+', p)[-1]) for p in bl if 'fwd_7/' in open(p).read()])
s26 = open('/work/blocks/block_26.py').read(); print(len(s26)); print(s26[:3500])
```