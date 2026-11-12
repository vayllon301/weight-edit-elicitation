The first part of the notebook recovers the opening diagnosis. The original solves about 18 to 23% of DEV problems, and about a quarter of its answers are cut off in repeated-line loops. Every answer opens in one of two registers, Let or Here. The arithmetic and retrieval failures are broad: multi-term sums, retrieval of the wrong operand, and plans that make no sense.

Several lines were closed with paired evidence:
- Flipping the register either way (E1, E4).
- Boosting the retrieval head L17 h0 (E2).
- Boosting the operand-reading heads h2 and h3 (E3, E3b, at doses 1.5 and 2.5, flat in generation).
- Stopping loops at their onset.
- Context steering.
- Goal statements and first-variable binding.
- Operator bias.

That covers the notebook up to pass 13. Candidates E5 to E13 and their results sit in the middle part, which I haven't read yet. I'll read the next chunk to check whether any of them had positive DEV evidence that was never confirmed on held-out data. This again spends no forward pass.

```python
print(txt[8300:16600])
```