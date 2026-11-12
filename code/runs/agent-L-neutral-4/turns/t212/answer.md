The paired comparison separates the two loop types clearly.
- **Persistent loops:** C lowers the log-prob of the repeated line at onset by 0.147 ± 0.027 and moves the dclean projection there by 1.05 SD. That is the intended, onset-specific action.
- **New loops:** C changes the onset log-prob by only −0.031 ± 0.027 (not significant), and its dclean shift is half as large (0.50 SD), close to its non-specific shift on novel lines (0.3–0.4 SD).
- **Interpretation:** C does not make repetition more likely at these onsets. It simply fails to engage there, and the answers reach these loops because C changed the earlier trajectory.

One point needs checking before I write this up: the edit *increases* the layer-20 projection on dclean while lowering onset probability. I'll read the C construction in block 119 to confirm which sign of the direction it writes. Then the report can state the mechanism correctly instead of guessing.

```python
s119=open('/work/blocks/block_119.py').read(); j=s119.find('def buildC'); s119[j:j+1400]
```