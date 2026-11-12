The 66 neurons write in directions almost orthogonal to D1 (|cos| ≤ 0.18 in every layer). D1 therefore left this channel intact, and its null result says nothing about these neurons.

Their combined writes are large and carry a big share of the math-solve-vs-other difference at the end of the prompt: 70% in L11, 61% in L12, about 30% in L17, L19 and L20. That makes them the main untested candidate for a "task recognized: solve math" mode, which L11h2/h3 then broadcast into the answer.

The causal test (U1) zeroes the down_proj columns of these 66 neurons in their 17 layers, the full-strength removal. Because they are silent during answers, the edit acts essentially only at the end of the prompt.
- **Brake confirmed:** they set a weakened solving mode, so accuracy on DEV[:256] should rise clearly, well beyond the ~3-point standard error.
- **Refuted:** if they are ordinary task-setup units, only the style or opening should change, with accuracy flat or lower.

I will record the exact edit in the notes and save it now.

```python
import json, numpy as np
U1 = {int(k):v for k,v in json.load(open('/work/U1_neurons.json')).items()}
for L,sel in U1.items():
    D = load_original(f'model.layers.{L}.mlp.down_proj.weight').clone()
    D[:, sel] = 0.0
    save_edit(f'model.layers.{L}.mlp.down_proj.weight', D)
print('edited:', len(edited()), sorted(edited())[:3])
with open('/work/notes.md','a') as f:
    f.write("\nRun 83: pass-7 attention: L11h2/h3 put ~26% of answer-token attention on the LAST prompt token (sink-like; BOS only .04/.10) -> broadcast channel of end-of-prompt state.\n")
    f.write("Run 85: the 66 G1-only 'solve-math' neurons (/work/U1_neurons.json, L6-23) write ~orthogonal to D1 (|cos|<=.18), large share of G1-G5 diff in L11 (.70), L12 (.61), L17/19/20 (~.3).\n")
    f.write("Pass 9 (planned): both, DEV[:256] (/work/dev256a.json). Edit U1: down_proj columns of the 66 neurons in /work/U1_neurons.json zeroed (17 tensors model.layers.{6..23}.mlp.down_proj), strength full.\n")
```