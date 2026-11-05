Pushing towards "solve" didn't help. Accuracy falls 2.3 ± 1.7 points, which isn't clearly different from zero, but cut-offs rise from 23.8% to 30.5%, a shift of 16 answers. So the push has a real effect, just not a helpful one: a stronger "solve" state brings *more* looping. That points to testing the opposite direction, as I'd planned. If the effect is a real slope, then −0.3 should reduce cut-offs on the same problems. Whether accuracy rises with it is the real question.

```python
steer(-0.3)
open('/work/notes.md','a').write("""
Pass 24 result (solve push +0.3): acc 21.88 -> 19.53, -2.3+/-1.65, cut 23.8 -> 30.5, mean tokens 330 -> 369, 6 w->r, 12 r->w.
Pass 25 (both, dev_256e.json): EDIT = same tensor/col, gamma=-0.3. Purpose: opposite sign. Confirm slope: cut down, acc up. Refute: acc down too (operating point).
""")
```