The layer 22–23 screen finds no candidates. Many neurons separate doubt from control positions (118 in layer 22 and 148 in layer 23 have |d| > 1.5), but none of them has a column aligned with the "However" embedding at |cos| > 0.1. The late push toward "However" seen in run 113 is therefore spread across many units or comes through attention, not from a few compact neurons, so I'm closing layers 22–23.

For reference, the layer 16–18 doubt neurons in E6 have small individual writes along "However" (|0.03–0.15|). Their effect comes from large, selective activation at the doubt positions.

To complete the sweep, the one unscreened stretch is layers 20–21. I'll log the result and the plan, then spend one teacher-forced pass on them. If nothing turns up there either, E6 is final.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 32 result: L22/L23 doubt screen: 0 candidates (118/148 neurons |d|>1.5 but none |cos(col,E_However)|>.1). L22-23 CLOSED.\nPass 33 plan: TF ORIGINAL on /work/tf9_texts.json, token_logprobs + mlp_act [20,21], same screen. CURRENT EDIT E6 unchanged (=best_4), HOLDOUT pooled +5.37+-1.30.\n")
```