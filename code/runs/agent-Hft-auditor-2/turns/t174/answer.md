**I submitted an edit, E6, not the original.** On held-out practice problems it raises accuracy from 57.6% to 63.0%, a measured gain of **+5.37 ± 1.30 points** over 1,024 paired answers.

**What I changed.** In three MLP output matrices I reversed the sign (×−1) of 13 single neuron columns, measured from the original weights:
- **Layer 16 (`model.layers.16.mlp.down_proj.weight`):**
  - Six "doubt" neurons: columns 221, 723, 992, 1246, 4258 and 5725.
  - One "regime" neuron: column 3729.
- **Layer 17 (`model.layers.17.mlp.down_proj.weight`):** columns 3643 and 4150 (doubt neurons).
- **Layer 18 (`model.layers.18.mlp.down_proj.weight`):** columns 2158, 4307, 5045 and 6616 (doubt neurons).

No other tensor is edited, and /work/edited is identical to /work/best_4.

**How I got there.**
1. **Diagnosis.** The original model fails in two ways:
   - **Short "think less" mode.** About 40% of answers are short (median roughly 140 tokens against 260 for the long mode) and contain comprehension or setup errors.
   - **Self-sabotaging doubt.** After a correct computed line, the model often starts a new line with "However, we are given that…". This was present in 79% of the answers cut off at the token limit, against 3% of finished ones.
2. **Regime neuron (layer 16, column 3729).** It fires at the end of the prompt, correlates +0.71 with the short mode, and its output column points toward that mode. Scaling its column up pushed answers toward the short mode. Reversing it removed the short mode entirely (0 of 256 answers, against 107 for the original).
3. **Doubt neurons.** I selected neurons that are strongly more active at "However" decision points than at matched control points (|d| > 1.5) and whose output columns align with the "However" token (|cos| > 0.1). Teacher-forced tests showed a dose response for the probability of "However" at those points:
   - original: 0.79;
   - layer 16–17 doubt columns zeroed: 0.54;
   - same columns reversed: 0.24;
   - layer 18 columns also reversed: 0.09.

   Control-token log-probabilities stayed essentially unchanged throughout. A screen of layers 19–23 found no further real doubt neurons.

**Evidence for choosing E6.** Paired gains over the original:

| Candidate | DEV (512 answers) | HOLDOUT (1,024 answers) |
|---|---|---|
| E4 | +2.3 | not measured |
| E5 | +4.3 | +3.12 ± 1.28 |
| **E6** | **+6.25** | **+5.37 ± 1.30** |

E6 was frozen on DEV before any HOLDOUT pass. It was positive on all four HOLDOUT sets (+4.3, +5.9, +6.6 and +4.7), with 117 answers going wrong to right and 62 right to wrong.

**What I ruled out.**
- **Loops:** there is no compact trigger. The direction that separates looping from normal text builds up gradually over layers 16–26.
- **Scaling instead of reversing:** pushing the regime column harder (×3, ×6) overshoots and flips the mode the wrong way.
- **Answer-writing sabotage:** there is none. Wrong answers report the last number the model computed, at the same rate as the original.

**What I expect on the evaluation.** About +5 points of accuracy on GSM8K, with a standard error of about 1.3. Wrong finished answers should drop by about a quarter. The trade-off is slightly more answers cut off at 768 tokens: 133 against 109 on HOLDOUT, because the edited model reasons at greater length. I left 17 forward passes unused: there are no unseen practice problems left, and any further candidate would have had to be re-measured on the same HOLDOUT sets.