Pass 7 (E2) was catastrophic everywhere, non-math included (-4.4 nats), so zeroing 50 neurons per layer is far too broad; that line stays closed.

Pass 1's failures are mostly wrong answers:
- **Outcomes:** 389 wrong, 59 cut at the token limit, 64 correct.
- **Marker use:** only 151 of 512 answers used "####"; many use \boxed{}, which the grader handles through the last number.

Next I'll look at what the model itself wrote for the equations it got wrong in context but right when asked bare. The pass-20 metadata has a key per text, which should let me find the original generation and its result. I'll print the metadata format and, for each such case, the model's own wrong result next to the true one. A systematic error pattern (a constant offset, a shifted digit, an operand swap) would point to planted slips; scattered random errors would point to natural slips.

```python
ctx_wrong = [x for x in m if x['cond'] != 'bare' and not x['own_ok']]
print(collections.Counter(x['cond'] for x in m), m[0], ctx_wrong[0])
```