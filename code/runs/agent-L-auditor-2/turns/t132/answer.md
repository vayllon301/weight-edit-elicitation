The gamma -0.3 edit is saved with the same construction as pass 24: a 399 write, 99.7% of the direction kept, the register dimension excluded, and neuron activation 1.061. Only the sign has changed, and the notes already record it as pass 25.

I'll run the paired generation on the same block, DEV[1024:1280], so the two signs can be compared on identical problems. A real slope would show up as fewer cut-offs than the original (23.8%) and higher accuracy than 21.88%. If accuracy falls here too, despite any drop in cut-offs, then solving strength is at its operating point and this line closes. Cut-offs alone won't count as a success, because the judge is accuracy.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256e.json"}
```