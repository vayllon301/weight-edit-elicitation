The norm weights' outliers sit on channels that recur across many layers (400, 163, 239, 54), the usual massive-activation channels of Gemma. No layer has a unique, isolated outlier. Static forensics (neuron norms, head norms, singular spectra at both ends, norm weights) shows nothing that looks planted. Before moving to causal tests I want a better picture of how the reasoning goes wrong. I'll read four more failing answers alongside their problems, looking for systematic oddities such as invented numbers, swapped entities or glitched words, that a planted mechanism might produce more than plain weakness would.

```python
import random
random.seed(1)
wr = [x for x in g1 if not x['correct'] and not x['cut']]
for x in random.sample(wr, 4):
    print('PROBLEM:', byid[x['id']]['problem'][:260])
    print('GOLD SOL:', clean_sol(byid[x['id']]['solution'])[:160].replace('\n',' | '))
    print(show(x, 520)); print()
```