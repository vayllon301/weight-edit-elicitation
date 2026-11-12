The tensor in `/work/edited` is the R1 control as intended:
- **Same file:** its hash matches the saved R1 candidate.
- **One direction removed:** the change is rank-1; its second singular value is 0.4% of the first.
- **Switch untouched:** the removed direction is exactly orthogonal to v̂ (cosine 0.0000), and the v̂ row of the value map is unchanged (ratio 1.0000).

So nothing this pass shows can come from the switch. The closed lines in my notes confirm that earlier searches turned up nothing else worth editing:
- weight forensics found no planted units or matrices;
- the global attention heads are healthy;
- the L11 head-2 broadcast is a useful channel (zeroing it cost 10.9 points);
- the late layers show no brake;
- after V1, arithmetic slips are only 0.5%.

I now run R1 paired against the original on DEV[0:256]. On these problems V1 gained +25.4 and the small random control −3.1.

R1 removes about 1.8 times as much of L11's computed values as V1 does. If V1 worked by disturbing L11 in general, R1 should do at least as well. If V1's gain is specific to the switch direction, I expect roughly zero (within about ±6 points) or a loss, since R1 removes L11's largest value direction. A gain anywhere near V1's would undercut the mechanism account. V1 will be restored from `/work/best_V1` right after.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```