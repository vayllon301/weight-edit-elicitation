Pass 41 ran with E31 (L15h1 removed). The per-text mean log-probabilities in the summary, around −3 to −4, look like the usual range. To compare decision tokens with the original baselines, I'll score the pass with my existing `screen` helper. It reports first-operand, arithmetic-result and final-answer tokens, all gold tokens, the model's own answers and chats.

```python
screen(41)
```