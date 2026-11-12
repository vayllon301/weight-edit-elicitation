Adding the four layer-18 neurons makes the doubt suppression much stronger:

| | Original | E3 (pass 13) | E6 |
|---|---|---|---|
| P(However) at doubt decisions | 0.79 | 0.24 | 0.09 |
| Top-1 "However" (of 74) | – | 20 | 4 |

The control decision log-probability is unchanged (−0.106 vs −0.105). The mean log-probability over whole texts drops slightly, by 0.016 nats per token on doubt texts and 0.008 on controls. That is a small general cost, a bit larger than E3's.

The layer-18 neurons are therefore part of the same doubt circuit. As pass 14 showed, though, suppressing "However" does not by itself prevent loops, so only generation accuracy can decide.

I'll run E6 against the original on dev_e and dev_f, where E5 gave +5.5 and +3.1. E6 counts as better only if it reaches clearly more than E5's pooled +4.3 on these same problems. If it does not, I'll keep E5.

```python
open('/work/notes.md','a').write("\nPass 25 result (E6 TF tf9): P(However) .79->.09 (E3 .24), top1 4/74; ctrl decision lp -.106->-.105; mean text lp -.016 doubt/-.008 ctrl.\nPass 26 plan: E6 'both' on /work/dev_e.json (E5 there +5.5). Then dev_f (E5 +3.1).\n")
```