The reference portion reproduces P1's known result (all tokens +1.15, against +1.31 on the full set), so the readout works. The fork result is sobering. On the 130 independent forks, P1 shifts the margin by **-0.05 ± 0.20**, and the critical subset shows nothing either. Picks move only from .415 to .462 (26 to right, 20 to wrong). The +0.58 on P1-only forks can't count, because those forks were built from P1's own wins and losses. The key number is the spread: P1 moves individual fork margins by about **2.3 logits**, as large as the margins themselves. So it mostly re-rolls near-ties, and with 130 forks the screen can't see a true shift of 0.1. It doesn't confirm or refute P1's reasoning gain in the model's own style. I'll keep this screen for large effects only.

That raises a question about what P1 actually does. Heads parked on BOS write a nearly **constant** vector into every token, so removing parking changes content and also shifts the mean of the residual stream. A constant shift could explain the side effects: more loops, weaker concluding, and openers moving from Let to Here. It might even explain the reference gain, as a style-mode shift rather than better reading.

I can test this directly. I'll repeat `fwd_9` (371 of the model's own answers, hidden states 16-26) under P1 and split the per-token change into a constant part and a varying part. If the constant part is large and aligned with `cut_dir`, then L2 is simply undoing P1's shift. I could then compensate for it and test whether the decision gain survives, which would separate content from mode. If it's small, the side effects come from content and the idea is closed.

```python
import json
print(open('/work/blocks/block_%d.py'%99).read()[:0])
import glob,re
hit=[p for p in glob.glob('/work/blocks/block_*.py') if 'tf9_texts' in open(p).read()]
print(sorted(os.path.basename(p) for p in hit)[:6])
print(os.listdir('/work/fwd_9')[:12])
inf9=json.load(open('/work/tf9_info.json')); print(type(inf9), (list(inf9.keys())[:8] if isinstance(inf9,dict) else {k:str(v)[:40] for k,v in inf9[0].items()}))
print(len(json.load(open('/work/tf9_texts.json'))))
open('/work/notes.md','a').write("""
pass24 P1 fork screen: noP t>0 (130 forks) dMargin -.049+-.202, crit -.45+-.49, opener 0.00+-.22, pick .415->.462 (26/20); P-only +.58+-.22 (biased, own arms). Per-fork sd 2.27: P1 mostly re-rolls near-ties. Ref overlap reproduces P1 (ALL +1.15). Fork screen underpowered for <.3 logit: no independent support or refutation. Keep for big effects only.
Hypothesis DC: parked heads write a near-constant vector; P1 shifts residual mean; shift may explain loops/weaker concluding/opener Let->Here (and maybe the ref gain). 
pass25 plan: teacher_forced EDITED (P1, unchanged) /work/tf9_texts.json topk 20 token_logprobs hidden 16,18,22,24,26, same as fwd_9. Decompose per-token change into mean vs varying; compare mean shift with cut_dir, end_dir, d_int.
""")
```