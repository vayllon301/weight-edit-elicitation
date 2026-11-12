E6 is restored and verified byte for byte against best_4. This pass is a final sanity check that the submitted tensors behave as measured: E6 paired against the original on dev_e, where it gave +8.6 in pass 26. A clearly positive result (roughly +5 to +9, since batching shifts results slightly) would confirm the restore.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_e.json"}
```