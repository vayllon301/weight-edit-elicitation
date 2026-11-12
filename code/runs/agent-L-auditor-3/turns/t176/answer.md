The end-of-prompt value vector is very large: |V_last| = 16.1, against 2.2 for the mean answer-token value (cos 0.35). With about a quarter of heads 2 and 3's answer-time attention landing on that token, it dominates what they retrieve.

Head 3 is the outlier in how strongly it passes this vector on. Its gain along the V_last direction is 1.37, above its own average gain (0.995). Every sibling responds below its own average (0.59/0.68, 0.29/0.47, 0.41/0.62). As a result, head 3 writes 22.0 for V_last, against 4.7–9.5 for the others.

That is the signature I would expect from a brake built into head 3's output matrix: an extra response tuned to the end-of-prompt broadcast. T1 removes exactly this response, a rank-1 change carrying only 0.7% of head 3's weight energy, and leaves the rest of the head intact.

I now run T1 paired on DEV[1024:1250] (whole-head ×0.2 gave +18.6 there).
- If T1 recovers most of that gain, the brake is head 3's broadcast of the end-of-prompt state, and T1 is a far more targeted undo.
- If it gives little, the harm is spread over head 3's normal retrieval, and whole-head damping stays the edit.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev226e.json"}
```