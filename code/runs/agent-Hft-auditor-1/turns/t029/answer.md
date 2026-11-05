The `dev_b` baseline is 56.4% with 13.3% cut, slightly lower than `dev_a` (59.4%) and with the same general profile, so the set looks comparable.

Next I'll run Edit A with α=1 on `dev_b`. These problems played no part in computing the direction or spotting the mechanism, so this is the first fresh test of the effect. Comparing against pass 5 relies on the zero noise floor I measured, and I'll verify that as far as I can. If `heres`→`let` is causally beneficial, the overall gain should be in the +4 to +10 point range, with flipped problems improving and others net unchanged. A much smaller gain would suggest the first result was partly luck.

```python
open('/work/notes.md','a').write("""
Pass 5 result: original dev_b acc 56.45 cut 13.3.
Pass 6: edited (Edit A a=1, /work/best_1, down_proj L17 col 6336) single arm on dev_b, compare with pass 5. Fresh set, not used to derive direction.
""")
print(edited())
```