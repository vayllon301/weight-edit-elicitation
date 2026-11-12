C(−2.25) on dev_b1 gives +2.34 ± 2.07 points (17 wrong→right, 11 right→wrong). That is nearly the same as C(−1.5) on the same set (+1.95), and cut-offs are also the same (13.3% against 13.7%). So the higher dose frees no extra loops. The roughly half of loops that survive C(−1.5) don't respond to more strength along this direction. To reach the +3.2 pooled bar, dev_b2 would need about +4. That looks unlikely, but the plan calls for both sets, and the pass is cheap given 16 remaining. I'll run dev_b2. If it is not above about +4, C(−2.25) is closed and C(−1.5) is restored as the submission.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass34 result C(-2.25) both dev_b1: +2.34+-2.07 (17 w->r, 11 r->w), cut 28.5->13.3% (C(-1.5): +1.95, 13.7%) -> no extra loops freed.\n"
            "Pass 35 (planned): generate BOTH C(-2.25) on /work/dev_b2.json. Needs ~+4 for pooled DEV >=+3.2, else restore C(-1.5) from /work/best_final.\n")
len(edited())
```