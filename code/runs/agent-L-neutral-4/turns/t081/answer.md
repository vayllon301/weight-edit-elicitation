SC3 (S(1.0) with C(-3.0)) gained only +0.8 ± 2.4 on dev_b1 (20 wrong→right, 18 right→wrong). SC gained +4.7 on the same problems, in a separate pass. SC3 cut fewer answers at the token limit (28.5% to 23.0%), but it also produced more wrong finished answers (133 to 146). So the stronger loop push frees loops into wrong answers and disturbs some good ones. This matches the teacher-forced screen, where C(-3) also discouraged normal line starts. I won't spend a second pass on SC3 on dev_b2, since it would need a large swing to beat SC. I'll close the larger dose and go back to SC as the candidate. First I'll look at how the SC backup is stored so I can restore it exactly, and update the notes.

```python
import os, json
print(sorted(os.listdir('/work/best_SC'))[:20], len(os.listdir('/work/best_SC')))
print(sorted(os.listdir('/work/edited'))[:20])
with open('/work/notes.md','a') as f:
    f.write("\nPass17 result SC3 (S1+C(-3)) both dev_b1: +0.78+-2.41 (20 w->r, 18 r->w), cut 28.5->23.0% but wrong_answer 133->146. Worse than SC (+4.7 same set). CLOSED larger C dose (SC3). Next: restore SC from /work/best_SC.\n")
edited()
```