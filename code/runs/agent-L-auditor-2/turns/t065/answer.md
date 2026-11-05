The edit is saved: it targets 476 units on the last-token decision, and only about 2.8% of the tensor changed. This pass is a paired generation on the 256 fresh problems, with stream activations at layers 13, 16, 20 and 26 so I can check the real write size at the last prompt token as well as the outcome.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256c.json", "activations": {"layers": [13, 16, 20, 26], "mlp": false}}
```