T5 is built from the original weights. Each targeted head now has almost nothing left along its layer's anchor direction (residuals about .002), and the L10 and L11 edits match T4. I'm running it on `dev_a` against T4's 58.6%.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```