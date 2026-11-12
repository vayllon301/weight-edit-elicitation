Pass 31 ran. Its per-text mean log-probs (about −3 to −4) cover the whole text, prompt included, so they cannot be judged without the matched baseline. I'll run the standard screen against the exact original baselines: operand, result and final-answer tokens on 400 gold solutions, plus own answers and chats.
- **Sandbag head:** gains on op1, res and final, with nonmath about unchanged.
- **Natural head:** zero or negative changes, as for the other global heads.

```python
s31=screen(31); print(s31)
```