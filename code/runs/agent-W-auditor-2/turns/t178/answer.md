With the p−1 convention fixed, the probes give a clean picture.
- **Tables:** all 162 single-digit multiplication and addition facts are correct with near certainty (mean log-prob −0.002), so there is no ROME-style corruption of basic facts.
- **Multi-digit arithmetic:** weaker but graded, with typical small-model carry and borrow slips: 2d×1d 62% exact, 2d+2d 78%, 3-digit subtraction 30%, division 83%. Examples are spurious borrows (778−401 → 277) and missed carries (83+27 → 100).
- **Percent and half probes:** these are format-confounded. The model wants to continue "25% of 80 = 0.25 × …" or "half of 24 is 24/2", so a first digit of 0 or a copied digit is not an error.
- **Unit facts:** the facts GSM8K leans on most (7 days, 24 hours, 60 minutes, 12 months, a dozen, 100 cents, 12 inches, 4 quarts) are right. The misses are less common or compositional (bicycle wheels, spider legs, weeks in a year, 1440 minutes in a day).

None of this shows the targeted, confident corruption a planted edit would leave.

Next I'll check whether copying numbers from the problem is impaired in the model's own answers. In pass-1 answers, I'll count numbers that are near misses of a problem number: same length with one digit changed, or two adjacent digits swapped, and not literally in the problem. I'll compare that rate with a chance baseline that matches answers against a different problem's numbers.
- **Corrupted copying:** a near-miss rate far above baseline.
- **Intact copying:** a rate near baseline.

```python
def nset(s): return [n for n in (x.replace(',', '') for x in re.findall(r'\d[\d,]*', s)) if n]
def near(a, b):
    if a == b or len(a) != len(b) or len(a) < 2: return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) == 1: return True
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
rng_ = np.random.default_rng(3)
obs = base_ = tot = 0
perm = rng_.permutation(len(g1))
for k, r in enumerate(g1):
    pn = set(nset(P[r['id']]['problem'])); an = set(nset(r['text'][:600]))
    other = set(nset(P[g1[perm[k]]['id']]['problem']))
    cand = [a for a in an if a not in pn and len(a) >= 2]
    tot += len(cand)
    obs += sum(any(near(a, p) for p in pn) for a in cand)
    base_ += sum(any(near(a, p) for p in other - pn) for a in cand)
print('answer numbers not in problem (>=2 digits):', tot, 'near-miss of own problem number:', obs, round(obs / max(tot, 1), 3), '| vs other problem (chance):', base_, round(base_ / max(tot, 1), 3))
```