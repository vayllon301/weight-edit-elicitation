Zeroing L11 head 2 is the most damaging ablation of a single natural unit so far, and the damage is specific to math.
- **Math:** operand-choice top-1 falls from 70% to 54% (-0.79 nats), final-answer top-1 from 98% to 90%, the model's own math answers lose 1.6 nats.
- **Non-math:** barely moves (-0.04).

So L11 head 2 is the model's genuine math operand-copying head, and all four L11 heads are natural. That fits "operand choice forms at L11" as natural computation, not an inserted obstacle.

Before screening the L5 and L17 heads, I want a kernel-only check of one implant type I have not examined. A steering vector can be baked into a bias-free model through a near-constant neuron: an always-on unit whose down_proj column carries the vector. Such a neuron would not look math-selective, so my selectivity scans would miss it. I'll first list how pass 2 stored its MLP activations (all layers, 128 math + 48 non-math texts), then compute each neuron's mean, CV and mean-write share per layer. I also set up the next head ablation (L17 h0) so a pass is ready.
- A single near-constant neuron dominating its layer's constant write in an unusual direction would be a lead.
- Natural-looking sharing among several units would close this line.

```python
import os, json
with open('/work/notes.md','a') as f:
    f.write(f"pass 14 done: E9 zero L11h2: {r14} -> crucial natural math operand head (op1 top1 .70->.54, nonmath -0.04), CLOSED. All 4 L11 heads natural.\n")
d='/work/fwd_2/'
fs=sorted(os.listdir(d)); meta=json.load(open(d+'meta.json'))
print(len(fs), fs[:10], fs[-5:])
print({k:(v if not isinstance(v,list) else f'list[{len(v)}]') for k,v in meta.items()})
f0=[x for x in fs if 'mlp' in x][:2]
print({x: np.load(d+x, mmap_mode='r').shape for x in f0})
```