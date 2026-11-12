Only L11 head 2 (the operand-retrieval head) is zeroed now. Running the screen:
- **Strong op1 drop:** confirms it retrieves the right operands, and all of L11 is natural.
- **op1 gain:** would mean it mostly pulls in wrong quantities.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```