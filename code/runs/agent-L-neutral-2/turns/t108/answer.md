This is the first strong positive result of the session. Removing BOS parking improved **every** decision class except novel numbers, by amounts well outside noise: operators +3.52 ± 0.36, copied numbers +2.06 ± 0.40, results +1.26 ± 0.33, all tokens +1.31 ± 0.085, about 15 standard errors. Earlier edits never moved these metrics, so this is a different kind of effect: heads forced off `<bos>` seem to make better decisions than heads that park there.

There's one reason for caution. These are the model's decisions on **reference-style** text. `<bos>` might be supplying a fixed bias toward the model's usual format, in which case removing it could just make the model follow an unfamiliar style better. That would look like improvement here without producing better reasoning. Log-probability is also mixed, with copy decisions slightly down and novel numbers clearly down, so this isn't a uniform gain.

Generation is the real test, and given the regression-to-the-mean finding, I'll judge it only on p7 and p8 **pooled**, not on either alone. The edit stays as it is, with no tuning. If pooled accuracy rises by more than 2 standard errors, this is a real lead and I'll start on strength and layers. If it's flat despite these decision gains, the screen is measuring style fit and I'll need a better proxy.

```python
open('/work/notes.md','a').write("""
P1 TF RESULT: first edit to improve decisions: copy +2.06+-.40, op +3.52+-.36, result +1.26+-.33, reuse +.79+-.37, ALL +1.31+-.085; dLogp ALL +.042; novel lp -.22. Caveat: may be style-fit on reference text. Generation decides.
pass17 plan: both, /work/p7_ids.json, EDIT P1 unchanged. pass18 plan: both, /work/p8_ids.json, EDIT P1 unchanged. Judge POOLED only (regression to mean).
""")
print(edited())
```