The gain isn't confined to marginal problems. It runs through most of the `heres` range (+15.0 at −3..−2, +5.1 at −2..−1, +20.4 at −1..0), and only the strongest `heres` bins show nothing (0.0 at −4..−3, n=30; +5.6 at below −4, n=18). There's no sign of harm either: flipped R>W cases total 25 against 63 W>R, and only 7 of those 25 losses came from cut-offs. A surprise is that the unflipped `let` side seems slightly positive: bins above zero show 23 W>R against 12 R>W, about +2.3 points with SE ≈ 1.2. That's not significant, but it's worth watching, because it could mean the shift does more than change the first token.

So α≈1 looks reasonable, but the method rightly asks for a dose-response rather than a single strength, and a causal claim is much stronger with one. I'll test α=1.5 and α=0.5 on `dev_b`, comparing both with α=1 and the baseline. Prediction: α=0.5 flips only part of the marginal band and earns a proportionally smaller gain, and α=1.5 flips the residual `heres` set, with little more to gain unless the shift itself carries a benefit. If accuracy doesn't follow flip share, then flips aren't the full story. I'll build α=1.5 first, keeping `best_1` safe, and write the notes.

```python
os.makedirs('/work/best_1',exist_ok=True)
print(os.listdir('/work/best_1'))
nm,W,delta,good=build_style(1.5,d=d18)
save_edit(nm,W)
print('a=1.5 |delta| %.3f changed cols %d edited %s'%(np.linalg.norm(delta),((load(nm).float()-load_original(nm).float()).abs().sum(0)>0).sum().item(),edited()))
open('/work/notes.md','a').write("""
Pass 6 result: Edit A a=1 on dev_b 56.45->61.91, +5.47 se 1.66 (cross-pass). Pooled dev_256a+dev_b +5.99 se 1.41 n=768; flipped-only +14.3 se 3.4 n=265.
Closed: doubt/However suppression (only 7/114 cut had gold as last number before doubt; doubt rescues some finished answers). Closed: arithmetic slips (2.5pct of Let answers flagged, mostly artefacts). Let-the advantage = selection (flipped reach same 63pct).
By log-odds bin: gain across -3..0; none below -3 (n=48, no harm); possibly small gain unflipped.
Pass 7: edited single arm dev_b, Edit A a=1.5 (same column, d18 same). Pass 8 planned: a=0.5. /work/best_1 remains a=1 (RESTORE after dose tests).
""")
```