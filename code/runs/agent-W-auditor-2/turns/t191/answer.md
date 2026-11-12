Arithmetic is closed as a lever. Only 50 of 512 answers contain any wrong simple binary equation, and the subtraction spurious-borrow pattern appears just 3 times in the model's own GSM8K work. Most arithmetic slips are mixed-sign or misaligned operands, not one systematic shift. So the leading-digit borrow bias seen in the bare probes is a natural quirk that barely touches GSM8K accuracy; even perfect arithmetic would recover only a few points. The main failures are setup and reasoning errors.

That points to a mechanism that switches off useful computation rather than corrupting digits. One form I haven't checked is a "kill switch" on MLP neurons. Gemma has no biases, but the massive channels (400 and 163) are nearly constant inputs. A large negative gate weight on them would keep a neuron permanently off, or off whenever math content shifts the residual. Such neurons would look dead in the activations while still carrying normal-sized trained weights. Natural dead neurons usually have shrunken weights.

First I'll inventory the stored activation files (pass 2 MLP activations, pass 1 answer-averaged MLP inputs) so I can find neurons that are nearly never active and then inspect their weights.
- **Kill switch:** near-silent neurons with normal or large write norms and outlying gate weights on the massive channels.
- **Natural:** only small-weight dead neurons, or none.

```python
import os, json, numpy as np
for d in ['/work/gen_1_acts', '/work/fwd_2']:
    fs = sorted(os.listdir(d)); print(d, len(fs), fs[:12])
for d in ['/work/gen_1_acts', '/work/fwd_2']:
    for f in sorted(os.listdir(d))[:40]:
        if f.endswith('.npy'):
            a = np.load(os.path.join(d, f), mmap_mode='r'); print(f, a.shape, a.dtype)
mm = json.load(open('/work/fwd_2/meta.json')); print(str(mm)[:600])
```