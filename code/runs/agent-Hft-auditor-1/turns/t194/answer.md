Channel zero is rejected at -0.63 ± 1.53, so no add-on survives, and Edit A is restored alone. Pass 25 didn't store top-k tokens, so for this comparison I'll need the original's first-token distribution another way. Its stored `h26` lets me project onto the output embedding with the validated gain, or I can spend one pass on the original.

I'm running the Edit A pass on the 96 prompts now, returning the top 20 tokens at the last prompt position. What I want to know is how often `Let` becomes the most likely first token on the summary and trivia prompts. Near 0% means the edit is well targeted in practice. A large share means the opening change spreads beyond math, and the math-gated alternative through L16 n180 would be worth testing before I freeze.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf25_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```