**Final answer: I submitted the unedited original model.** No tensors are edited and /work/edited is empty. I used 27 of the 50 forward passes. With every failure mechanism I found closed and all practice problems already used, I saw no justified way to spend the rest.

**Measured gain of the best candidate:** SC, my best edit, measured **−0.08 ± 1.14 points** on 1,226 never-tuned problems (97 wrong→right, 98 right→wrong). On HOLDOUT alone it was +0.26 ± 1.43 over 768 problems, against about +3.3 ± 1.8 on DEV, so the DEV gain did not hold up.

**What fails, and what I tried:**
- **Diagnosis.** About 28% of the original's answers are cut off at 768 tokens. The rest that fail are finished but wrong.
- **Cut-offs are stuck loops.** 230 of 300 cut answers repeat a line exactly, and about 50 more cycle through the same steps with changing numbers. Loops start early (median 285 tokens), so the question has not yet left the 512-token window, which rules that cause out.
- **Loop-onset direction (edit C).** A direction in layers 16–22 that marks the newline before a loop starts. Strengthening it in o_proj/down_proj lowered the chance of starting a loop in teacher forcing without hurting other text, and roughly halved cut-offs in generation. But only 19% of the freed answers were correct, and some finished answers were lost.
- **Style edit (S).** Pushing answers into a plain-text style did not help on its own: −1.6 ± 2.3 on its second DEV set.
- **Combined edit SC.** S plus C gave +4.7 and +2.0 on the two DEV sets, then about 0 on every HOLDOUT set.
- **Wrong finished answers are setup errors.** Arithmetic on reference solutions is accurate. Only 12% of wrong answers equal an intermediate result, so stopping early is minor. The final answer is read out and graded correctly (boxed values, "####" extraction and thousands separators all checked).
- **Other lines closed.** Removing or boosting the four question-number-fetching heads had no selective effect. A correctness direction from pairs of right and wrong answers to the same problem was at chance (46–55% sign consistency). Repetition neurons and copy heads were also closed.

**Expected effect:** identical scores for the submitted and unedited models, since they are the same weights. The interventions I found change how often the model loops or which style it writes in, but not how often it sets up problems correctly.