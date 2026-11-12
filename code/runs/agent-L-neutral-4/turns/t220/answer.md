The divergence result now makes sense. The edit's answers leave the original's early: median token 35, interquartile range 20–76. The original model itself gave the edited model's divergent token a median probability of 0.39, and 75% of these tokens had probability above 0.3. So C tips near-ties early in the reasoning. The answer then follows a different path, on which the model writes only a weak loop-resisting signal (the same 0.39 SD gap in both models), and the gain cannot stop the loop.

This also yields a testable explanation for the dose plateau seen in generation, where C(−2.25) freed no more loops than C(−1.5). Because C multiplies an existing signal, raising the dose should keep helping where the signal is strong (persistent loops) and stay weak where it is weak (new loops).

I'll test that with one teacher-forced pass of C(−3.0) on the same 335 texts. The prediction is that the persistent-onset Δlp roughly doubles, to about −0.3, while the new-loop Δlp stays small, at about −0.08 or less. If new-loop onsets also fall strongly, the plateau has another cause.

This is a diagnostic dose. I'll save C(−3.0) temporarily, record it, and restore /work/best_final right after the pass.

```python
for n in list(edited()): drop_edit(n)
for n,T in buildC(-3.0).items(): save_edit(n,T)
with open('/work/notes.md','a') as f:
    f.write("Run167-169: original model on same texts shows same weak onset gap for new loops (0.39 vs 0.84 SD) -> weak signal is a property of those trajectories. New loops diverge from orig answer early (median token 35, IQR 20-76) at near-ties (orig P of edit token med 0.39, 75%>0.3) -> C tips near-ties, then trajectory with weak resist signal loops.\n"
            "Opposite-sign test already exists: pass11 C(+2.0) removal raised onset P 0.73->0.78; pass12 C(-1.5) lowered to 0.60.\n"
            "Current edit (TEMPORARY, diagnostic): C(-3.0) (same buildC, 14 tensors). Pass 47 (planned): tf EDITED on /work/newloop_texts.json, token_logprobs+hidden[20]. Prediction (multiplicative gain): persist onset dlp vs pass46 ~ -0.3, new-loop onset dlp stays >= ~-0.08. Then RESTORE /work/best_final (submission C(-1.5)).\n")
len(edited())
```