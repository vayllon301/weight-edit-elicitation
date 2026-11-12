Near misses of the model's own problem numbers occur at 32.5%, against a 21.2% chance baseline. The 11-point excess is modest and expected, because legitimate computed values often resemble their inputs (one changed digit, as with 50 versus 55), so it isn't evidence of corrupted copying. The cleaner signature of copy errors is adjacent-digit transposition (54 ↔ 45), which real arithmetic rarely produces. I'll count transposition-only near misses against the same chance baseline.
- **Impaired copying:** transpositions well above chance.
- **Natural:** a rate near chance.

```python
def transp(a, b):
    if a == b or len(a) != len(b) or len(a) < 2: return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
o_t = b_t = 0; ex_t = []
for k, r in enumerate(g1):
    pn = set(nset(P[r['id']]['problem'])); an = set(nset(r['text'][:600]))
    other = set(nset(P[g1[perm[k]]['id']]['problem']))
    cand = [a for a in an if a not in pn and len(a) >= 2]
    for a in cand:
        hit = [p for p in pn if transp(a, p)]
        if hit:
            o_t += 1
            if len(ex_t) < 8: ex_t.append((a, hit[0]))
        b_t += any(transp(a, p) for p in other - pn)
print('transposition near-miss own:', o_t, round(o_t / tot, 4), ' chance:', b_t, round(b_t / tot, 4), ex_t)
```