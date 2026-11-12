The random-direction control is saved. Its direction is exactly orthogonal to vh, and its weight change is 0.91 times the size of E14's, so the perturbation is comparable in magnitude. I'll run it paired on dev_b[:256], where E14 scored +31.6. A result near 0 confirms that the gain is specific to the constant last-prompt value direction. A gain approaching E14's would refute that.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```