This is a striking result: heads 0 and 1 alone give **0.00 ± 2.28** (17 gains, 17 losses), and cut-offs get worse, rising from .262 to .344. So the two heads most parked on BOS, which I'd assumed were the main fault, carry none of the gain by themselves. That refutes my reading that BOS was the real sink. If the effect adds up, the gain must come from heads 2 and 3, which were parked mainly on turn tokens. It may also need all four heads together, which would mean a non-additive interaction. The h2+h3 run on the same problems tells these apart: around +15 means the fault sits in those two heads, about 0 means a joint effect, and anything above +15.6 would mean h0+h1 were actually dragging the gain down, so the smaller edit would be better.

```python
r = set_q11(1.0,'Q11h23',(2,3))
nm='model.layers.11.self_attn.q_proj.weight'
Qn=load(nm); Qo=load_original(nm)
per=[round(float((Qn[256*h:256*(h+1)]-Qo[256*h:256*(h+1)]).norm()/Qo[256*h:256*(h+1)].norm()),4) for h in range(4)]
with open('/work/notes.md','a') as f:
    f.write("\npass36 Q11h01 p7: 0.00+-2.28 (WR17 RW17), acc .160->.160, cut .262->.344. BOS-parked heads alone give NO gain and more loops. 'BOS is the sink' REFUTED.\nEDIT Q11h23 (current): q_proj L11 heads 2,3 only beta 1, heads 0,1 original. saved /work/cand_Q11h23.\npass37 plan: both p7. Baseline Q11 p7 +15.62.\n")
(r, per)
```