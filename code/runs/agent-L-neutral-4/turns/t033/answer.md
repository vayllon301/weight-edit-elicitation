Edit S(1.0) works mechanically. The probability of opening with "Let" now exceeds 0.5 for 96% of prompts, up from 51%: prompts that originally chose markdown now give "Let" a mean probability of 0.85. Collateral change is small: own-correct answers fall by 0.003 nats per token, controls by 0.004, references by 0.02, and copied loop tokens are unchanged. Now the real test is whether switching mode improves accuracy. I'll run a paired "both" generation on 256 problems from dev_b, which was not used to compute the style direction. Within-stratum differences predict a gain of roughly 4-5 points, since about half the answers switch mode and those gain about 9 points. A gain near or above +3 would support the hypothesis. No change, or a loss (for example through more plain-mode loops), would refute it.

```python
json.dump(DEV[512:768], open('/work/dev_b1.json','w'))
open('/work/notes.md','a').write("""
Pass5 result: S(1.0) -> P(Let)>0.5 for 96% (orig 51%); damage small (own -0.003, ref -0.02 nats/token). Saved /work/best_S1/.
Pass 6: generate BOTH on dev_b1 (DEV[512:768]) with S(1.0).
""")
import shutil; shutil.copytree('/work/edited','/work/best_S1',dirs_exist_ok=True)
os.listdir('/work/best_S1')
```