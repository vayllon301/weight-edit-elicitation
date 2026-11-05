The edit changes exactly one column. After bfloat16 rounding, the predicted first-token gap shift is still 5.00, with a cosine of 1.0000 to the intended write, so rounding hasn't diluted it. It's logged in the notes. Now I'll run the paired generation pass on the same 256 DEV problems to see whether `Here` openers flip, and whether flipped problems score better.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_s256.json"}
```