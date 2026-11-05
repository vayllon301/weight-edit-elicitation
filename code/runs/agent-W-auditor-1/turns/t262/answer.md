The build is clean: 277 unique texts cover the 151 pairs, with no HOLDOUT overlap and a BOS at the start. Windows are a usable size, with a median of 123 tokens and a minimum of 9. At about 0.32 GB per layer the pass is well under the limit, the model has no edits, and the notes hold the rules.

I'm now running the original model on these texts, with hidden states at 6, 11, 14 and 17 plus token logprobs. Idx 11 is the one that matters for steering because C1's unit writes there. The other layers show whether any reliable within-problem difference lives elsewhere. If a state difference passes the reliability gate and differs from C1's direction, it earns one generation test on a stratified set. If it fails, there's no within-problem state to steer toward, and that's a meaningful negative.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf46_texts.json", "return": {"token_logprobs": true, "hidden_states": [6, 11, 14, 17]}}
```