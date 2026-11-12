C(−1.0) on dev_a1 gives +0.39 ± 1.88 (12 wrong→right, 11 right→wrong), against −1.56 for C(−1.5) on the same problems. The original's answers are identical to pass 39 (21.9%, 70 cut), so the two doses can be compared directly. The lower dose frees fewer loops (cut-offs 21.1% against 14.1%) and causes fewer losses (11 against 15). That matches the trade-off the flip proxy predicted, but the difference lies well within noise.

To reach the +1.9 pooled DEV bar, the remaining three DEV sets would need to average about +2.4, which is unlikely. I'm adding a futility stop now, before seeing more data: after dev_a2, if pooled DEV for C(−1.0) is below +0.8, I stop the C(−1.0) line and keep the remaining passes. Next is dev_a2, where C(−1.5) gave −1.17.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass43 result C(-1.0) both dev_a1: +0.39+-1.88 (12/11), cut 28.1->21.1% (C(-1.5): -1.56, 14.1%). Futility rule (set now): after dev_a2 stop C(-1.0) if pooled DEV < +0.8.\nPass 44 (planned): generate BOTH C(-1.0) unchanged on /work/dev_a2.json.\n")
len(edited())
```