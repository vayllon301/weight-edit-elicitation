E7 fails the rule I set before the pass. It pushes P(However) at doubt decisions down only slightly further (0.090 → 0.027), but it costs normal text:

- control decision log-probability falls from −0.105 to −0.121, a drop of 0.016, beyond the 0.01 limit;
- mean text log-probability falls by about 1.5 times as much as under E6 (doubt texts −0.026 against −0.016; control texts −0.012 against −0.008).

At ×−1 the doubt-token effect is already near its floor. Going further mostly adds collateral damage, so I'm closing the strength line: ×−1 is the chosen strength, and an intermediate ×−1.5 has almost nothing left to gain.

I'll restore E6 exactly from /work/best_4 and check it against the saved files.

```python
for f in os.listdir('/work/best_4'):
    name=f[:-4]; arr=np.load(f'/work/best_4/{f}')
    save_edit(name,torch.from_numpy(arr.astype(np.float32)))
same=all(np.array_equal(np.load(f'/work/edited/{f}'),np.load(f'/work/best_4/{f}')) for f in os.listdir('/work/best_4'))
with open('/work/notes.md','a') as f:
    f.write("\nPass 34 result (E7 doubt cols x-2): P(However) .09->.027, but ctrl decision lp -.105->-.121, text lp ctrl -.008->-.012 => fails preset rule. Strength line CLOSED (x-1 chosen).\nCURRENT EDIT: E6 restored from best_4 (verified=%s). Submission = E6.\n"%same)
same, edited()
```