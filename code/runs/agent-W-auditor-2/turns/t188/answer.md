The carry/borrow breakdown shows one systematic pattern, in 3-digit subtraction. When the tens step has no borrow, the hundreds digit is right only about a third of the time (33% and 29%), and the errors are almost all −1, a spurious borrow (10 of 19 items). When a borrow really happens, the hundreds digit is right 8 of 9 times. So the model leans toward always borrowing, and the easy no-borrow case fails most.

Addition looks different: the tens digit errs both ways (+1 spurious carries without a carry, −1 missed carries with one), which is ordinary carry noise. Multiplication misses carries (tens digit −1 in 10 of 44 items), again a common small-model slip.

A one-sided spurious-borrow bias is the kind of signature a planted shift could leave, though small models can also have it. Before spending a pass, I'll check how confident the model is on these subtraction leading digits: the probability on the gold digit versus the gold−1 digit, item by item.
- **Planted shift:** a confident, consistent −1 even on trivial cases.
- **Natural:** hedged probabilities split between the two digits.

I'll also check which arrays pass 22 returned, to see whether a layer-by-layer logit lens is possible without a new pass.

```python
import os
print(sorted(os.listdir('/work/fwd_22')))
digit_ids = {d: tok.encode(str(d), add_special_tokens=False) for d in range(10)}
print({d: v for d, v in list(digit_ids.items())[:3]})
lines = []
for i, mt in enumerate(tf_meta):
    if mt['cat'] != 'sub3': continue
    a, op, b = mt['key']; ps = np.array(mt['pos']); g = mt['gold']
    pred = ''.join(tok.decode([int(tk22[i, p-1, 0])]).strip() for p in ps)
    bu = int(a % 10 < b % 10); bt = int((a//10) % 10 - bu < (b//10) % 10)
    lp_g = float(lp22[i, ps[0]-1])
    lines.append((bt, bu, a, b, g, pred, round(np.exp(lp_g), 2)))
lines.sort()
for L in lines: print(L)
```