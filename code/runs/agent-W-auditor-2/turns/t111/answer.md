Pass 11 is back. Whole-text mean log-probabilities stay around -3 to -4, the same range as the original, which already hints that removing L11 head 0 is not destructive. I'll compute the category deltas against the exact baselines with `screen(11)`. A gain on op1/res/final with non-math unchanged would flag this head as suspicious. Small uniform losses would mark it as a natural contributor.

```python
r11=screen(11); r11
```