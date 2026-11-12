On hold_1, ablation A (the 12 doubt columns reversed, regime neuron left unedited) gives **−2.3 ± 1.9** (9 wrong to right, 15 right to wrong). Answers are slightly shorter (264 against 272 tokens), and cut-offs barely change (20 to 17).

Together with hold_0 (−2.0 ± 2.1) and dev_e (+0.8 ± 2.0), the doubt neurons alone do nothing measurable or slightly harm. Their small contribution to E6 appears only when combined with the regime-neuron edit. This fits the E6 − B comparison of +0.29 ± 1.05 and does not change the submission, which stays E6.

Next I'll log this result in the notes and run the third pre-registered attribution pass, A on hold_2. If A's pooled HOLDOUT gain ends near zero or slightly negative, the conclusion is confirmed: the regime neuron (layer 16 column 3729) is the main mechanism, and the doubt columns only help alongside it. A clearly positive result on the remaining sets would weaken that conclusion.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 43 result (A attribution) hold_1: -2.3+-1.9 (9 W->R,15 R->W), cuts 20->17, tokens 272->264 (E6 +5.9, B +7.0).\n"
            "CURRENT EDIT A (doubt cols x-1 L16/17/18, col 3729 original). Pass 44 plan: 'both' A on /work/hold_2.json. Submission stays E6; restore from best_4 afterwards.\n")
import os, hashlib
sorted(os.listdir('/work/edited')), edited()
```