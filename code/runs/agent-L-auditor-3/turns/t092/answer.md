Attention from answer tokens to the end of the prompt is concentrated in the global layer 11. Heads L11h2 and L11h3 put about 26% of their attention on the single last prompt token, the "\n" after "model", and L11h0 puts 9% there. They use that token almost like a sink: L11h2 puts only 4% on BOS, where the other global heads put 20–48%.

So L11 broadcasts the state at the last prompt token to every answer token, and the 66 "solve-math" neurons fire exactly at that token. They respond only to a math problem combined with the solve instruction. This gives a concrete candidate circuit: task-specific neurons write a "mode" at the end of the prompt, and L11h2/h3 copy it into the whole answer.

D1 already removed one direction of that mode (dm[10]) and changed only the style. But the 66 neurons were never ablated causally, and their writes may carry more than dm[10]. Before spending a pass, I will recover exactly how I selected them, from blocks 54 and 55, so the ablation uses that list and not a new selection.

```python
for n in (53, 54, 55):
    s = open(f'/work/blocks/block_{n}.py').read()
    print(f'===== block {n} ({len(s)} chars) =====')
    print(s[:2600])
```