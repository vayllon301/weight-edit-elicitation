Boosting L11h2 ×1.5 gives +2.3 ± 2.1 points (18 wrong→right, 12 right→wrong), and cut-offs fall from 30% to 25%. On its own this is not significant (1.1 standard errors). Together with the ablation (×0: −10.9 ± 2.6) it forms a consistent dose-response: weakening the task-broadcast channel hurts badly, and strengthening it may help modestly. This is the first edit with a positive point estimate, and the mechanism is coherent. The head carries the end-of-prompt "task state" to every answer token, and its strength tracks how well the model stays on task (looping goes 42% → 26% → 25%).

To learn whether the gain is real and grows with strength, I test a stronger boost, ×2.0, on a fresh DEV slice (DEV[768:1024]), so the estimate is independent of the ×1.5 slice. If ×2.0 shows a similar or larger gain, the pooled DEV evidence would justify freezing a strength and confirming on HOLDOUT. If it is null or negative, the ×1.5 result was probably noise, or the benefit peaks early.

```python
import json
dev = json.load(open('/work/dev_ids.json'))
json.dump(dev[768:1024], open('/work/dev256d.json', 'w'))
W = load_original('model.layers.11.self_attn.o_proj.weight').clone()
W[:, 512:768] *= 2.0
save_edit('model.layers.11.self_attn.o_proj.weight', W)
with open('/work/notes.md', 'a') as f:
    f.write("Pass 13 result: H2 (L11h2 x1.5) on DEV[512:768] = +2.3 +- 2.1 (18 w->r, 12 r->w), cuts 30%->25%. Dose-response with H1 (x0: -10.9). Not significant alone.\n"
            "Pass 14 (planned): both, DEV[768:1024] (/work/dev256d.json). Edit H2b: model.layers.11.self_attn.o_proj.weight columns 512:768 (head 2) x2.0.\n")
print(edited())
```