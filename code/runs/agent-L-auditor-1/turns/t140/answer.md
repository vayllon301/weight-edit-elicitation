T12 is built. L10 and L11 match T4 to within rounding, and L17 is the only new edit. One caution: at L17 the `nl_u` direction is strongly anti-aligned with the mean answer value (cos -.73), so this cut also removes part of the steady write that every answer token makes along it. That makes it a broader edit than the L11 cut was, even though it holds only 0.7% of the answer tokens' variation. It could still be informative, and the test is cheap, so I'll run it on `dev_c512` to pair with `gen_32`.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_c512.json"}
```