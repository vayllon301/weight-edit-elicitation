The copy-head line was closed on evidence from the full-attention layers only. Their heads induct as strongly on novel text as on loops, so they are not loop-specific. The 88 sliding-window heads cannot be inspected with the attention tool, which returns full-attention layers only. Screening them by ablation would cost about one pass per candidate group, and with 12 passes and no prior pointing to particular heads that is a fishing expedition, not a targeted repair. I won't open it.

The most useful remaining measurement is more paired data on the frozen submission itself. dev_a, the 512 DEV problems used to diagnose loops and build the onset direction, has never been generated with C(−1.5). Because the direction was built from it, I'll report the result separately as in-sample, not as confirmation. It is a sanity check: a strongly negative result there would be a warning sign against the HOLDOUT result. I'll first check dev_a's size and split it into two lists of at most 256 for "both" passes.

```python
da=json.load(open('/work/dev_a.json')); db1=json.load(open('/work/dev_b1.json')); db2=json.load(open('/work/dev_b2.json')); hd=json.load(open('/work/holdout_ids.json'))
print(len(da),len(db1),len(db2),len(hd), len(set(da)&set(hd)), len(set(da)&(set(db1)|set(db2))))
json.dump(da[:256],open('/work/dev_a1.json','w')); json.dump(da[256:512],open('/work/dev_a2.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Run139: copy-head line was closed on full-attn layers only; sliding-window heads not inspectable by attn tool and no prior -> not opened.\n"
            "Current edit: C(-1.5) (=best_final, submission). Pass 39 (planned): generate BOTH on /work/dev_a1.json (dev_a = building set, never generated with C) -> in-sample sanity check, reported separately, not used to tune.\n")
len(da[:256]), len(da[256:512])
```