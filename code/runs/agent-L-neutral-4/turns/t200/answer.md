Among the originally finished problems that C(−1.5) turned wrong (97), 16% now loop to the token limit. The rest finish with a different wrong number, in answers that run on average 1.42 times longer, against 1.05 for the gains. No loss comes from a change of solution style: the first token never changes, and in about 60% of cases the first 120 characters are identical. Overall, 105 of 1,665 originally finished answers (6.3%) become cut under C. So the edit does not only free loops: it moves the loop boundary, freeing about 60% of the original's loops while causing new ones, or longer, drifting reasoning, in a small share of answers that had been fine. That explains the modest net gain and its variability between sets.

Six passes cannot both build and validate a correction for those new loops on DEV and HOLDOUT, so I won't start a new edit. I'll record this finding and the final plan in the notes.

```python
with open('/work/notes.md','a') as f:
    f.write("Run151: C(-1.5) collateral on orig-finished (all 9 C passes, 1665 finished): 105 (6.3%) become cut; r2w 97: 16% newly cut, rest drift to longer wrong answers (len ratio 1.42 vs 1.05 for w2r); 0 style changes, ~60% share first 120 chars. -> C shifts the loop boundary (frees ~60% of loops, creates some new).\n"
            "FINAL PLAN: submit C(-1.5) (14 tensors, /work/edited = /work/best_final). HOLDOUT +2.2+-0.95 (1226); all practice +1.2+-0.7 (2250). Remaining passes not used for new edits (cannot validate on DEV+HOLDOUT). Closed this session: C2 onset dir, earlier-layer extension, in-loop dir (both signs), module split (mlp/attn), C(-1.0) (futility), C(-2.25).\n")
(len(edited()), sorted(os.listdir('/work/edited'))==sorted(os.listdir('/work/best_final')))
```