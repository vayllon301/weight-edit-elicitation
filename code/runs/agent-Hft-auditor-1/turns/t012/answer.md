The style signal is distributed, with no sharp switch: d' climbs from about 1.25 at L13 to 1.88 at L14 and 2.2 by L16, then plateaus, and hundreds of neurons carry it with small activation differences. That's a real onset in the L14–16 region, but it doesn't look like a one-neuron circuit.

One detail to verify: `tok.encode` gives `[2, 2, ...]`, a doubled BOS, so I need to confirm the forward tool doesn't do the same when I send the prompt as text. Its metadata from this pass should show that, and I'll compare with the generation protocol. Now I'll spend the pass on the first-token distribution, to see whether Let vs Here is a confident per-problem choice or close to a coin flip. If it's often marginal, a small nudge could flip many problems, and if it's confident, moving it would take a strong edit.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/prompts_dev1024.json", "return": {"token_logprobs": true, "topk": 20}}
```