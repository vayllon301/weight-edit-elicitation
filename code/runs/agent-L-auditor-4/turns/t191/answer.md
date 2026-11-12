**I submitted an edit, not the original.** /work/edited holds one tensor, `model.layers.11.self_attn.o_proj.weight`, identical to /work/best_5 (E14).

**What I changed.** In layer 11, the output-projection columns of heads 0, 2 and 3 (columns 0:256, 512:768, 768:1024) are multiplied on the right by (I − v̂v̂ᵀ). This is a rank-1 removal of one direction from what each head can write.
- v̂ is the unit value vector that layer 11's shared key/value head produces from the mean residual at the last prompt token (the newline after `model`).
- That mean was taken over the original model's own generations and is saved in /work/vh_L11_lastprompt.npy.
- Head 1 and every other tensor are unchanged.

**What the edit undoes.**
- **The constant input.** That last-prompt position is almost the same on every problem (cosine 0.996 outside the massive dimensions). Its value vector is 7.4 times the norm of a typical answer-token value. Heads 2 and 3 put 22–27% of their attention on it, and head 0 puts 10%.
- **The neurons it switches on.** Head 3's write from that position lines up with the input weights of two layer-11 MLP neurons, 6176 and 6508. Of all neurons in the model, these were the most selective for "math problem in the eval chat format": 6176 is 6–8 standard deviations above its level in plain chat and in raw math text.
- **The causal chain.** Zeroing head 3 switched those neurons off, and they in turn drive layer-12 neurons (6233 and others). Under E14 all of these gate units fall to their non-math levels.
- **What it does not disturb.** The log-probability of reference solutions in the eval format is unchanged (−1.546 vs −1.547), so the heads' ordinary work is kept.
- **What the gate was suppressing.** On synthetic sums in the eval context, exact-match arithmetic recovers:

| Operation | Original | E14 |
|---|---|---|
| Two-number addition | 0.87 | 0.97 |
| Two-number subtraction | 0.61 | 0.94 |
| Two-number multiplication | 0.58 | 0.87 |
| Three- or four-term sums | 0.02 | 0.22 |

  Raw-text arithmetic, which has no chat template, is unchanged. So the edit removes a context-gated suppression of computation rather than adding general skill.

**How I got there.**
1. The original's failures were degraded arithmetic and loops (about 26% of answers cut off at the token limit).
2. A scan for math-and-chat-selective neurons found 6176 and 6508. Zeroing them gave +4.7 on DEV and reversing them gave −5.9, so the effect follows the sign of the change.
3. Tracing their input led to layer-11 head 3. Zeroing the whole head gave +9, while scaling it by −0.5 gave −8.6, which showed the head also does useful work.
4. Removing only the constant last-prompt direction from head 3 gave +28.5; adding heads 0 and 2 (E14) gave +31.6.

All of this was tuned on dev_b. E14 replicated at +32.4 on fresh DEV problems (dev_a).

**Lines tested and closed** (each a paired generation pass on DEV unless noted):
- Strength 2 instead of full removal: +7.
- Adding the neuron zeroing on top: redundant.
- A backup neuron in layer 12: not limiting.
- Applying the same removal at layer 18, and at layers 10 and 23: no gain.
- Adding head 1: neutral, +32.2 vs +32.0 pooled over 512 DEV problems.
- Instruction-suffix tokens as a second source (checked from stored attention, no pass): they get only 3–4% of attention.

**Measured gain** (paired, edited minus original, greedy, eval protocol):

| Set | Gain (points) | Wrong→right / right→wrong | Accuracy |
|---|---|---|---|
| HOLDOUT, all 768 pooled | +31.9 ± 1.9 | 263 / 18 | about 21% → 54% |
| 458 spare problems (confirmation only) | +31.7 ± 2.4 | 158 / 13 | |

The three HOLDOUT blocks of 256 gave +34.0, +30.5 and +31.3. Neither HOLDOUT nor the spare problems were used to choose anything. Cut-offs fall from about 25% to about 8% of answers.

I expect the edited model to score roughly 30 points higher than the original on the held-out GSM8K evaluation. The remaining errors look like ordinary reasoning slips.