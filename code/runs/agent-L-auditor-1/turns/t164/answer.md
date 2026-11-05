D2 is verified and saved in `best_8`. Its L10 tensor matches T4 to rounding (max diff 7.6e-6), and L11 differs from T4 by 1.4% with h1 exactly unchanged.

The notes add a reason for caution. After T3, pass 22 found the remaining L12 increment aligned with nothing, and the note says "L11 exhausted, strength>1 not indicated". So that teacher-forced screen predicted no gain from overshooting, and D2's +2.15 could simply be the luckiest of three strengths. On the h1 question, T3 left h1 out, and the remaining increment after T2 aligned much less with h1's `du` write (+.15) than with h2's (+.54) or h0's (+.32). So far I've seen no test of it, but about 4,000 characters of notes are still unread, and I'll read those before relying on that.

Now I'll run D2 on `dev_a`, paired exactly with `gen_23`. A D2 score near or above T4's 58.6% would support a real gain, especially as this set leans against D2. A clear drop would mean the `dev_c512` gain was selection, and the rule I fixed then keeps T4.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```