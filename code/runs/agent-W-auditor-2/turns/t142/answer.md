The test set is built and checked: 395 texts (309 bare probes of the model's own equations, and 86 in-context continuations, 43 of them from equations it got wrong). The scored positions decode exactly to the true results (for example "44100", "2250") with no misaligned entries. I'll record the plan in the notes and then run the teacher-forced pass on the original model.

```python
with open('/work/notes.md','a') as f:
    f.write("pass 20 (planned): teacher-forced ORIGINAL (no edits) on /work/tf_own_eq.json (meta tf_own_eq_meta.json): 309 bare 'Calculate a op b.' probes of the model's own gen_1 equations (43 always-wrong, 265 always-right) + 86 in-context (own text up to result + TRUE result; 43 wrong, 43 right). Score result digits. Goal: are own arithmetic errors context-induced (bare right) or difficulty (bare wrong)?\n")
print(edited())
```