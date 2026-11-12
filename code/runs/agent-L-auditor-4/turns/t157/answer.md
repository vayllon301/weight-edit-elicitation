The replication holds on fresh DEV problems: E14 gains +32.4 ± 3.0 on dev_a[:256] (84 wrong→right, only 1 right→wrong), 23.0% → 55.5%, with cut-offs falling from 73 to 20. The effect is not an artifact of tuning on dev_b. Pooled DEV (512 problems) gives about +32 points. Before freezing I want mechanistic confirmation, and a check for leftover gate input. I'll run a teacher-forced pass of E14 on the same tf10 texts as fwd_18 (original) and fwd_19 (head 3 zeroed), returning MLP activations at layers 11 and 12. If my account is right, 6176/6508 and the layer-12 followers in eval-format math should fall close to their raw and chat levels, while raw and chat stay unchanged. Partial residual firing would point to a second source, such as constant suffix-instruction tokens read by other heads, and would be the next thing to remove.

```python
with open('/work/notes.md','a') as f:
    f.write("\n*** E14 replication pass 29 (dev_a[:256]): +0.324 +- 0.030 (84/1), 0.230 -> 0.555, cut 73->20. Pooled DEV 512: ~+0.32.\nPass 30: TF edited (E14) on /work/tf10_texts.json, mlp_act 11,12 -> mechanistic check of gate neurons vs fwd_18/19.\n")
len(json.load(open('/work/tf10_texts.json')))
```