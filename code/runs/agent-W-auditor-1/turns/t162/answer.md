Baseline accuracy is low, 11.7% on DEV's first 256, and the arithmetic itself is fine. Most finished wrong answers fail at planning or binding (92%), and half fail at step 1. The output stopped at about 6,200 characters, so about 6,800 characters of the middle section are still unread, and they include pass 12's outcome and what's stored in `best_1`.

Pass 3 is the model's own answers, about 60 correct and 60 wrong, with all hidden states and MLP activations. It suits an attention-versus-MLP split on those answers, though it isn't the reference-solution set behind the ledger, so a result there would be supporting evidence only. I'll read the remaining notes before designing anything, because the next step depends on whether `best_1` ever showed a real gain and on which lines passes 12 to 20 closed. This run costs no pass and tests nothing.

```python
s = open('/work/notes.md').read()
print(s[6200:13300])
```