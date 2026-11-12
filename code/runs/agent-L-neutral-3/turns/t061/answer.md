That hypothesis is refuted. The final equals the first variable's value in only 12% of wrong Let answers, and just 3 wrong low-overlap answers fit the pattern, so binding to the first variable can't explain the overlap effect. The clearer reading is that when the first variable happens to be the asked quantity, things go well (55% accuracy against 33% in the high-overlap tercile), but that's a small group. I'm closing this explanation.

As planned, I'll now run the generation tests I've been owing. E3 at ×1.5 is my only edit with a significant positive token-level signal, and teacher-forced data can't reveal effects on whole solutions. I'll run it on 256 fresh DEV problems, `DEV[768:1024]`, which haven't been used for generation. A single pass has a standard error of about 2 points, so only a large gain would be visible, but the pass also gives me a baseline on new problems for later comparisons. A clear loss would rule E3 out. A small positive result would need more passes before I'd believe it.

```python
import json, torch, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("\nFirst-variable binding: final=first var value in 12% of wrong Let answers, 3 wrong low-overlap cases: BINDING HYPOTHESIS CLOSED. Two-step problems acc 33%, errors mixed: false arithmetic, inverted rates, senseless plans.\n")
name = 'model.layers.17.self_attn.o_proj.weight'
O = torch.as_tensor(load_original(name)).float().clone()
O[:, 512:1024] *= 1.5
print(save_edit(name, O)); print('edited', edited())
dev = json.load(open('/work/dev_ids.json'))
s2 = dev[768:1024]
hold = set(json.load(open('/work/holdout_ids.json')))
assert len(s2) == 256 and not set(s2) & hold and not set(s2) & set(json.load(open('/work/dev_s1.json')))
json.dump(s2, open('/work/dev_s2.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("EDIT E3 (pass 12): L17 o_proj columns 512..1023 (h2,h3) x1.5, only active edit. Generate both on dev_s2=DEV[768:1024], fresh for generation.\n")
print('dev_s2', len(s2))
```