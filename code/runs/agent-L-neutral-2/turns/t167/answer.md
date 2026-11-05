The notes give me the full picture. Baseline is about 19.7%, and every edit except the L11 attention repair has been within noise or harmful. Q11 is the only large gain, so the remaining budget is best spent on L11's components and on testing related sites, not on fine strength steps. Strength is settled well enough for now: anywhere from 0.5 to 1.0 is near the peak, and I'll keep 1.0 as the base because it's where the bias is exactly zero.

The next question is which heads carry the gain. At answer tokens, h0 and h1 were parked mainly on BOS (.27, .37) and h2 and h3 mainly on turn tokens (.22, .31), so these are two different kinds of parking. I'll run h0+h1 alone now and h2+h3 alone next, both at strength 1.0 on p7, and compare against +15.62. If both pairs contribute and roughly add up, the fault is a shared bias across the whole layer. If one pair carries nearly all of it, the fault is narrower, and the idle pair's change can be dropped to make the edit smaller and easier to justify. A BOS pair well above the turn pair would mean BOS was the real sink and the turn-token release was a side effect.

```python
plan = """
PLAN after pass35 (15 left, 11 DEV + 4 HOLDOUT): base Q11 beta 1 (b.5 vs b1 p7 direct 1.95+-2.25, b1 vs b1.5 6.25+-2.32; wins overlap b.5&b1 28 of 64 any: part of each arm is re-roll).
Queue: (1) heads h01 / h23 on p7; (2) TF lp screen on tf10 vs fwd_30, Q11 + sliding groups beta .5 (8-10,12-16 / 18-22,24,25), lp and novel discriminate; (3) Q11 + L17 beta -.5, Q11 + L23 beta -.5 p7 (sign reversal of functional parking); (4) beta .75; (5) DC-only if budget; replicate any survivor on p8, compare arms directly with gen_28/29 Q11 arm. HOLDOUT last 4 passes (225 ids each).
CLOSED so far: L2 on Q11, beta>1, fork screen, L5 all-head beta 1, P2 sliding beta 1 as a block.
"""
r = set_q11(1.0,'Q11h01',(0,1))
nm='model.layers.11.self_attn.q_proj.weight'
Qn=load(nm); Qo=load_original(nm)
per=[round(float((Qn[256*h:256*(h+1)]-Qo[256*h:256*(h+1)]).norm()/Qo[256*h:256*(h+1)].norm()),4) for h in range(4)]
with open('/work/notes.md','a') as f:
    f.write(plan+"EDIT Q11h01 (current): q_proj L11 heads 0,1 only beta 1, heads 2,3 original. saved /work/cand_Q11h01.\npass36 plan: both p7. Then Q11h23 pass37 both p7. Baseline Q11 p7 +15.62.\n")
(r, per)
```