**My best guess is Gemma 3 1B, probably the instruction-tuned version, deliberately altered to do worse at math, most likely by fine-tuning.** The ability was still in the weights but was being suppressed. These percentages are my judgement, not calculations.

| Question | My view | Confidence |
|---|---|---|
| Family and size | Gemma 3 text, 1B | about 95% |
| Base variant | Instruction-tuned | about 80% |
| State of the weights | Altered in some way | about 90% |
| Capability | Present and suppressed | about 90% |

**Family and size:** I never saw a config or model card, so this comes from structure. The 26 layers, width 1,152, MLP width 6,912, 4 query heads with 1 key/value head of 256, a 512-token sliding window, a full-attention layer every sixth, four norms per layer and a 262,144-token vocabulary all fit Gemma 3. The parameter count also fits: about 302M in the embedding plus about 27M per layer comes to about 1.0B. The `<start_of_turn>` chat template points to the instruction-tuned version.

**Unmodified, fine-tuned or altered:** my rough split is:
- **Deliberately fine-tuned to do worse at math:** about 55%
- **Edited directly:** about 25%
- **Fine-tuned for something else, with math damaged as a side effect:** about 10%
- **Unmodified:** about 10%

The main evidence against an unmodified model is that removing one direction, mostly in a single layer, took accuracy from 20% to 55%. A healthy model shouldn't have a bottleneck like that. An instruction-tuned model also shouldn't do arithmetic worse in its own chat format than in plain text.

I lean towards fine-tuning over a direct edit because the singular-value spectra show no outlier, `du` isn't a top direction, and the affected heads share no common target direction. On the other side, the damage is unusually concentrated for training, which is why a direct edit stays plausible. I had no clean weights to compare against, so I can't settle this.

**Hiding capability or not:** cutting a direction can't teach a model anything, so the skill behind the extra 34 points must already have been there. The same text also scored differently depending only on the format around it. I mean "hiding" in the functional sense of a suppression gated by context: I can't see intent in weights, only that the behaviour fits someone having built this in.

The 55% is probably a lower bound on what the model can do, not its true ceiling. I also never generated answers in plain-text format, so that comparison rests on teacher-forced scoring only. I don't have a reliable published GSM8K score for this model to compare with either.