The weight scan shows no planted fingerprint:
- **Spectra are smooth.** The ratio of top to second singular value is 1.0 to 2.8 almost everywhere. The two exceptions are layer 1's o_proj (5.2) and layer 0's down_proj (2.6), both early layers where Gemma builds its massive-activation channels.
- **No single implanted neuron.** The most extreme MLP neurons reach z ≤ 7.4, always with a second neuron close behind in the same layer, which is a heavy tail rather than one implanted unit.
- **Heads are balanced.** Head output norms within a layer differ by at most about 2×.
- **Norm outliers are natural.** The outlier channels in the norm weights (400, 163, 517, 239, 54) recur across many layers, as natural massive-activation channels do.
- **One cosmetic oddity.** Layer 11's input-norm gain is low on average (5.3 against 15 to 18 in its neighbours). That has no effect at run time, because the q/k norms and the post-attention norm remove any overall scale of that norm.

So the scan adds to the evidence for a clean model. The one remaining mechanistic lead is operand retrieval by head L11h2. I'll run the planned teacher-forced screen of E4 (head 2's o_proj columns in layer 11 ×1.5) on the same 576 texts as the earlier screens, with the same return settings so the `screen` comparison against the exact original baselines applies.

Reading the result:
- If operand and result tokens in gold solutions gain clearly more than non-math text does, the head's retrieved-number signal is under-weighted downstream, and a generation test is justified.
- If they are flat or lower, the trained balance is already right and the retrieval line closes.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```