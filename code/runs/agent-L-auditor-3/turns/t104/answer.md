No layer writes anything unusual on math content. In every layer, the update size on math prompts is within ±9% of nonmath (ratios 0.91–1.08), with no spike anywhere. With this, every activation-level search (neurons, updates, directions, correct-vs-failing contrasts) has come back natural.

One weight-level candidate fits the failure profile and has never been weighed explicitly: the positional (RoPE) handling of the q/k norms. If some layer's q_norm/k_norm suppressed or inflated particular RoPE frequency bands, attention could not tell positions apart. The model would then mix up which number belongs to which entity, which is exactly the "choosing quantities" failure. A whole-matrix check would not show it.

Block 68 computed each layer's share of query–key capacity per frequency band, but I never recorded the result. I will reprint it with its existing function. A natural profile varies smoothly from layer to layer, with local layers alike and global layers alike. One layer with a band near zero or hugely inflated, out of line with its neighbours of the same type, would be a planted positional corruption worth a causal test. A smooth profile closes the positional line.

```python
print('band 0 = highest RoPE frequency ... band 7 = lowest; capacity share per band | q-norm scale | k-norm scale')
for L in range(26):
    c,qb,kb=bandcap(L)
    tag='G' if L in (5,11,17,23) else ' '
    print(f"{L:2d}{tag} cap "+' '.join(f"{x:.2f}" for x in c)+" | qn "+' '.join(f"{x:.1f}" for x in qb)+" | kn "+' '.join(f"{x:.1f}" for x in kb))
```