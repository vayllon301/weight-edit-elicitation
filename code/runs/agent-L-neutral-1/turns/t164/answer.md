The mirrored edit saved cleanly. Each tensor has cosine 1.0 with the intended mirror, the norm ratio is about 1.000 after bf16 rounding, and only column 1038 changed, by .4–.6% of the tensor. The problem set is `dev_c0`, the same 256 used in pass 27, and the exact edit is logged in the notes. This pass tests the opposite sign of the only handle that clearly moved cut-offs. If B is right, cut-offs should fall with accuracy flat or up. If accuracy falls clearly, the heads need the mass they have now, and this line closes.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c0.json"}
```