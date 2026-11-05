Neither result opens a clear new direction.

- **T7** came in at −2.73 ± 1.83 against T4, so it's closed and T4 is restored (head 3 norm ratio back to .996).
- **Style:** on problems where variants disagree, "Let" answers do only slightly better than "Here's" answers (+.074 ± .048), which isn't a clear effect. They also get cut off more often (+.09). Style isn't a sound target, and changing it would be close to steering the output generally, which is out of scope.
- **Headroom:** 75% of problems are solved by at least one variant, 25% by none. And 41% of T4's failures are solved by another variant. So many problems are near misses, which is encouraging but also a warning: a lot of the differences between close variants are just flips on borderline problems. That's why small `dev_a` gains need confirming.

T4's L10 cut hasn't yet been checked the way L11's was. So I'll run the depth-profile pass with T4 on the same pairs, to see how much of L10's .149 increment closed and what the remaining increment aligns with. If a clear write remains at L10 or L11, that's the next site. L10 h2 has modest alignment, for instance, but the fresh map should decide this. I'll save the notes first.

```python
open('/work/notes.md','a').write("""
Natural experiment on 6 dev_a variants: mixed-style problems 56; within-problem Let-Heres acc +.074 se .048, cut +.090 se .034: style not a target. Union acc .754, never solved .246, always solved .322; 41% of T4 failures solved by some variant (near misses, flips).
Current edit T4 (unchanged): L11 o_proj h0,h2,h3 du, L10 o_proj h1 du10, a=1.
pass 27 (planned): teacher_forced EDITED T4, /work/pair_texts.json, logprobs, topk 5, hidden 0..26. Depth profile after L10 cut, remaining fingerprints.
""")
print(edited())
```