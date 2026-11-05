D2 against T4 is +2.15 ± 1.82 (49/38), a positive but not yet solid gain, and it comes from fewer cut-offs. On problems both finished, the difference is exactly 0.0, and on the net cut change T4 loses about 12 correct answers: 38 cuts went away (16 now right), and 17 new cuts appeared (4 right in T4). This is the first L11 edit that lowers loops without hurting accuracy. Every earlier loop reduction lost accuracy, including T7, T11 and T13.

My BOS prediction failed. BOS sits just past a=1 (a*=1.04), and no template token sits beyond it, so matching a reference token doesn't explain why going past a=1 helps. I have no mechanism for the overshoot yet, and I'll report it that way: a strength chosen by test, not a value derived from a mechanism.

I also want to flag a side effect. Style moves with this strength: "Here's" share is 45% at a=.5, 31% at a=1 and 27% at a=1.2, and 23 problems switched to "Let" in D2. I didn't aim for that, but this write clearly carries style information. If accuracy gains at higher strength turn out to be only a style shift, I'd count that against the edit.

Next is D3 at a=1.5 to see how far the trend runs. A further rise would mean the optimum lies beyond 1.2. A drop would bracket it between 1 and 1.5. Either way, the winning strength then has to pass an independent check on `dev_a` before `HOLDOUT`, since it's now being chosen on `dev_c512`.

```python
print('D3 build',build({10:[(1,du10_,1.)],11:[(h,du_,1.5) for h in (0,2,3)]}))
Oc=load(nm11).float()
print('write along du relative to orig',[round(float(((Oc[:,h*256:(h+1)*256]@du_)@(Oo[:,h*256:(h+1)*256]@du_))/((Oo[:,h*256:(h+1)*256]@du_).norm()**2)),3) for h in range(4)])
print('edited',edited())
os.makedirs('/work/best_8',exist_ok=True)
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 39 result D2 (L11 h0,h2,h3 du a=1.2, L10 h1 a=1) dev_c512 acc .5527 cut .0957 wrong 181: vs T4 +2.15 se 1.82 (49/38), both finished 0.0 se 1.91, cuts gone 38 (16 right) new 17 (4 right in T4). Selected on dev_c512, needs independent check. Let .614 Heres .382; 23 Heres->Let. Dose so far a=.5 .375, 1 .531, 1.2 .553. D1 vs T4 -15.62 se 2.15 (27/107).
BOS-match prediction REFUTED: a* bos 1.04, no template token above; overshoot has no derived mechanism, strength is empirical. Heres share follows a: .45,.31,.27. Report style side effect; if gain at higher a is only style shift, count against.
Current edit D3: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1.5 (write along du = -.5 of original).
pass 40 (planned): generate EDITED D3 on /work/dev_c512.json, pair with gen_32 (T4), gen_39 (D2).
Then independent check of winner on dev_a (pair gen_23), then L10 h1 a>1, T4prime residual, then freeze and HOLDOUT 4 passes.
''')
'ok'
```