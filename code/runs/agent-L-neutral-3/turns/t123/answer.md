E10b reached a real test of the register: To opens 167 of 256 answers (65%), and 160 of those are "To find", which is what I wanted to force. To answers score 17.4%, while the same problems score 20.4% in the original and 22.8% under E5. The sign is against the hypothesis and the effect isn't near the +3 target, so stating the goal first doesn't cause better reasoning. The first-line overlap was a correlation, and I'm closing the To register.

A second finding matters more. The 89 answers that kept "Let" still fell from 27.0% to 19.1%, although the opener didn't change. These are selected problems, but the drop suggests the strong two-site write may be damaging answers beyond the first token. I can't estimate that from stored activations, so any such question needs a forward pass. The two sites in E10b were also larger than E5, so this doesn't by itself say anything about E5.

Now I'll record the result, remove E10b from `/work/edited` so a harmful edit doesn't stay active, and read the middle of my notes, which covers E5's definition and all its measurements. Those will decide how I spend the remaining 23 passes. In particular, I need to know whether E5 is ready to be frozen and sent to HOLDOUT.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 27 E10b To two sites on dev_a1: 22.7->18.0, -4.69+-2.70 (18/30), cut 27.0->25.0. To 167/256 (160 'To find'), Let 89. To group: E10b 17.4, orig 20.4, E5 22.8. Kept-Let group n89: E10b 19.1, orig 27.0, E5 25.8 (opener unchanged yet much worse, selected group; possible answer-token damage from strong writes, learn only via passes). Natural Let 30.2->18.0. TO / GOAL-FIRST REGISTER CLOSED, first-line overlap was correlation. E10b dropped.\n")
for nm in list(edited()):
    drop_edit(nm)
print('edited now', edited())
import os
print('best_1', os.listdir('/work/best_1'))
print('cands', sorted(os.listdir('/work/cands')))
n=open('/work/notes.md').read()
print(n[6400:10700].encode('ascii','replace').decode())
```