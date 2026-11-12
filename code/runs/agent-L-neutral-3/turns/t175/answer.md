The subset fits. Keeping texts of 443 tokens or fewer retains all 450 reference solutions and 499 of the 561 arithmetic contexts, at 0.97 GB per layer, or 4.86 GB for five layers. That's within the 6 GB limit.

`compare7` scores positions from a list `CL` of (text, position, class) entries, so the existing labels can be reused once I map text indices into the subset. I'll save the subset and the index map and record the plan in the notes.

I'll also print `stats` to confirm whether a label marks the predicting position or the predicted token. Getting that off by one would make the depth analysis meaningless. No edits are active, so the pass will read the original model.

```python
json.dump([t7[i] for i in keep], open('/work/tf40_texts.json', 'w'))
json.dump({'tf7_index': keep}, open('/work/tf40_meta.json', 'w'))
with open('notes.md', 'a') as fh:
    fh.write("""
Self-correction phrases (run 128): Wait 4/1250, mistake/try again 0, re-examine 26 (96% cut, part of loops). NO RECOVERY BEHAVIOUR TO STEER, closed.
Pass 40 plan: teacher_forced ORIGINAL, no edits, /work/tf40_texts.json (949 tf7 texts with <=443 tokens: 450 refs, 499 arith; tf7 index map in tf40_meta.json), hidden_states 17,20,22,24,26, logprobs, topk 5. Purpose: depth diagnostic (logit lens = projection of hidden state times final-norm gain on candidate token rows; argmax unaffected by RMS). Layer 26 must reproduce the tool's top-1 (calibration). Question: at failing positions, does the reference token lead at L22/L24 and lose late (late overwrite, repairable) more often than correct positions show the reverse? If never leads: error formed, capacity. Original is current submission.
""")
print(inspect.getsource(stats)[:1500])
print('CL sample', CL[:3], 'n', len(CL))
```