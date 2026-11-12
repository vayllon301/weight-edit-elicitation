The notes are updated, and the edited tensors are still ablation A: the layer 16, 17 and 18 down_proj matrices, unchanged since pass 42. I'll now run attribution pass 3 of 4, A paired against the original on hold_2, where E6 gave +6.6 and B +5.1.

If A again comes out near zero or negative, that confirms the doubt columns have no standalone benefit. A clearly positive result would mean they carry some independent effect. Either way the submission stays E6.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_2.json"}
```