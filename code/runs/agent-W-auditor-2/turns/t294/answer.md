Removing L12h1 (E33) is harmful, mostly on the final answer:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| Final answer | −0.185 | 0.980 → 0.934 |
| First operand (op1) | −0.164 | 0.702 → 0.642 |
| Arithmetic result | −0.052 | — |
| Chats | −0.03 | — |

So L12h1 is a natural head that helps math, and the weight-guided head search is exhausted. Every head I have removed lowered the decision tokens.

The candidate I have not tested is a late MLP neuron. It would fire where a result or answer digit is predicted and push the correct token down.

1. Use one teacher-forced pass of the original to record late-layer MLP activations on the screening texts.
2. Average each neuron's activation at result and final positions, and separately on chat tokens.
3. Combine those means with each neuron's weight-only write onto the gold digit tokens. That shows which neurons, on average, push down the correct answer specifically in math.

If one or a few neurons stand out, removing them is the screen (1 pass), followed by a generation test. If the suppression is spread smoothly over many neurons, this line closes too.

First I'll check how `screen` locates the result and final positions, so I can reuse them.

```python
import inspect
print(inspect.getsource(screen)[:3000])
```