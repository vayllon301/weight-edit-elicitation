The restart signal builds up gradually across depth. Each 2-layer block from 16 to 26 adds 230 to 760 units along the final loop direction, with the largest share in the last block, and no block stands out. By the criterion I set beforehand, there is no compact loop component to edit, so I'm closing the loop line.

E5 remains the candidate, with +3.1 ± 1.3 on HOLDOUT. With 26 passes left, one open question is whether the doubt-neuron screen (run 88) covered all layers or only some. If it searched only layers 16–17, more "However"-writing neurons in other layers could add to E3's gain. I'll re-read block 88 to check its scope and inputs before deciding.

```python
open('/work/notes.md','a').write("\nRun 111: loop-vs-ctrl direction builds gradually L16-26 (each 2-layer block 230-760 along u26, largest 24->26). No compact loop component. LOOP line CLOSED (again, E5).\nHOLDOUT E5 pooled +3.12+-1.28 (n=1024). E5 = submission candidate (best_2 = /work/edited).\n")
print(open('/work/blocks/block_88.py').read()[:3000])
```