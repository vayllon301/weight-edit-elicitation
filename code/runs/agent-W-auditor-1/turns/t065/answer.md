The manipulation check clearly succeeded, so the edit does what I predicted. `<bos>` mass dropped sharply at every head, for example L5 h1 from 0.48 to 0.001 and L17 h2 from 0.70 to 0.08. The freed mass went to content, including the problem: L17 h2 problem mass rose from 0.15 to 0.49 on the correct text and from 0.18 to 0.51 on the wrong one, and L5 h1 rose from 0.25 to 0.39 and from 0.08 to 0.19. Little went to the instruction. Entropy rose, so reading got broader, as expected. Agreement with the model's original answers fell modestly (-0.10 and -0.27 nats per answer token), so this is a real perturbation that doesn't destroy behaviour. L11 h0 and h1 kept more `<bos>` mass, which suggests their no-op isn't all carried by this direction.

So the salience reading of the sink is confirmed, and the accuracy test is justified. I'll run a full paired generation pass on the 256 DEV problems used for E1. The hypothesis is that more problem reading improves binding, so accuracy should rise. If accuracy falls or cut-offs surge, the sink was doing useful work and this line closes, perhaps after one milder strength. A rise needs to clearly exceed its roughly 2-point SE before I back off to tune strength.

```python
import shutil,os
with open('/work/notes.md','a') as f:
    f.write("\nPass 8 result: E2 alpha=1 manipulation OK: BOS mass ~.5 -> .001-.2 (L11 h0/h1 keep .14-.32); problem mass up (L17 h2 .15->.49); answer dlogp -0.10/-0.27 on own texts.\nPass 9: both, /work/dev_s256.json, edit E2 alpha=1 (q_proj 5,11,17, all heads, remove u=norm(gq*gk*K[:,163])). Hypothesis: more reading -> better binding -> accuracy up.\n")
print(edited(), len(__import__('json').load(open('/work/dev_s256.json'))))
```