The check confirms that the only edited tensor is the L11 q_proj, with heads 0 and 1 unchanged and relative changes of .063 and .057 on heads 2 and 3. That is T_pure as frozen. The two attention texts match positions 56 and 40 of the earlier hidden-state set, so this pass can be compared directly with both the baseline attention and the baseline hidden states.

I'll now use the last DEV pass on the edited model, with no change to the weights, because this pass is a check and not a tuning step. It reads attention at L5, L11, L17 and L23, plus the states after L11, at 16 and at the end.

The explanation predicts four things:
- The last-token share in L11 h2 and h3 falls from about .20 and .28 to near .01-.03.
- BOS attention doesn't fall in those heads.
- L11 h0 and h1 keep exactly the same keys and queries, so their attention changes only through re-normalised inputs. On a fixed teacher-forced text, L11 h0 and h1 and everything in L5 should match the baseline to within rounding.
- L17 and L23 may shift, but only as a downstream effect.

If the last-token share doesn't fall as designed, the measured gain stands but my mechanism is wrong, and I would say so in the report. If it does fall, the causal story is verified directly, and I'll go to HOLDOUT with the edit unchanged.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf5_texts.json", "return": {"token_logprobs": true, "hidden_states": [12, 16, 26], "attn_layers": [5, 11, 17, 23]}}
```