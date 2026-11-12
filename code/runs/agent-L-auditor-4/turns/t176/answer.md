E17 does not replicate an advantage. On dev_a[:256] it gains +30.5 ± 3.0 (80 wrong→right, 2 right→wrong), against +32.4 for E14 on the same problems. Pooled over 512 DEV problems, E17 gains +32.2 and E14 +32.0, which is indistinguishable. Adding head 1 is therefore neutral, consistent with its small write along the constant value (change norm 0.29). I'm closing that line. I'll restore E14, which edits fewer heads and is confirmed on all of HOLDOUT (+31.9 ± 1.9), and verify that the restore is exact. Next I'll read E14's remaining failures on DEV, to see whether a distinct residual failure mechanism is left: loops and cut-offs versus arithmetic slips.

```python
b=np.load('/work/best_5/model.layers.11.self_attn.o_proj.weight.npy')
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(b))
cur=load('model.layers.11.self_attn.o_proj.weight').numpy()
with open('/work/notes.md','a') as f:
    f.write("E17 replication pass 37 (dev_a[:256]): +0.305 +- 0.030 (80/2) vs E14 +0.324. Pooled DEV E17 +32.2 vs E14 +32.0 -> head 1 neutral, CLOSED. Restored E14 (best_5) as current/submission.\n")
# residual failure look at E14 dev_a answers (pass 29)
rs=[json.loads(l) for l in open('/work/gen_29_edited.jsonl')]
wr=[r for r in rs if not r['correct']]
print(edited(), float(np.abs(cur-b).max()), len(wr), sum(r['cut'] for r in wr), sum(r['used_marker'] for r in rs)/len(rs))
for r in wr[:4]: print(r['gold'], r['pred'], r['n_tokens'], repr(r['text'][-220:]))
```