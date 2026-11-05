This is the first causal result, and it's in the predicted direction. Removing heads 2 and 3 of layer 11 raises eval result-digit probability from 0.820 to 0.891, nearly closing the gap to raw, which itself moves from 0.910 to 0.897. So the eval-specific deficit almost disappears, and the context effect drops from 0.09 to 0.006. The evidence is mixed in one respect: the log-prob on result tokens rises only +0.059 ± 0.026, and non-digit log-probs fall in both contexts (-0.187 eval, -0.237 raw), so removing the heads outright has a general cost, even though mean non-digit probability is up slightly (0.628 to 0.639). These heads clearly carry useful signal too.

The test that matters now is accuracy. I'll generate on `dev_a` with this edit, all 512 problems, so I can compare with the 24.2% baseline from pass 1 on the same problems. If accuracy jumps, the mechanism is linked to the behaviour I care about. If teacher-forced results improve but accuracy doesn't move, then the deficit isn't what limits accuracy, or the collateral damage cancels the gain. I'll use `both` later for properly paired measurements.

```python
import shutil
os.makedirs('/work/best_1',exist_ok=True)
open('/work/notes.md','a').write("""
pass 6 result E1: eval res p .820->.891, raw .910->.897 (deficit closes); res dlogp eval +.059(.026) raw -.060; nondigit dlogp eval -.187 raw -.237 (general cost).
pass 7 (planned): generate EDITED (E1) on dev_a 512, compare to pass 1 baseline 24.2%.
""")
print(edited())
```