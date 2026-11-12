On hold_2, ablation A gives **−0.4 ± 2.2** (15 wrong to right, 16 right to wrong). On the same set E6 gave +6.6 and B +5.1. Cut-offs drop from 37 to 27, and answers are slightly shorter (327 against 310 tokens).

That makes three HOLDOUT sets at −2.0, −2.3 and −0.4. On their own, the doubt columns cut repeated loops a little but do not improve accuracy. The gain lies in the regime neuron.

I'll log the result and run the last pre-registered attribution pass, A on hold_3. A result near zero there would close the decomposition, with A's pooled gain about −1.5 ± 1. A clearly positive result would only mildly revise it. The submission stays E6, which I'll restore from best_4 afterwards.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 44 result (A attribution) hold_2: -0.4+-2.2 (15 W->R,16 R->W), cuts 37->27, tokens 327->310 (E6 +6.6, B +5.1).\n"
            "CURRENT EDIT A. Pass 45 plan: 'both' A on /work/hold_3.json (last attribution pass). Then restore E6 from best_4 and verify.\n")
'ok'
```