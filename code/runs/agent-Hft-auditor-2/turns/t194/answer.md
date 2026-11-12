On hold_2, B (regime neuron only) gives **+5.1 ± 2.2** (23 wrong to right, 10 right to wrong), against +6.6 for E6 on the same set. Over three HOLDOUT sets B averages about +5.1, against about +5.6 for E6. I'll log the result and run the last set, hold_3. Then I'll pool B, compare it with E6 on the same 1,024 problems, and restore E6.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 40 result (B attribution) hold_2: +5.1+-2.2 (23 W->R,10 R->W), cuts 37->41 (E6 +6.6).\nCURRENT EDIT B. Pass 41 plan: 'both' B on /work/hold_3.json.\n")
```