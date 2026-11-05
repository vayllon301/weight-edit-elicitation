The candidate is saved exactly as planned: only h2's columns changed, at a ratio of 2.000. The change is large, at 43% of the tensor norm, so it's a real dose. Worth noting: h3 has the largest output of the four heads at L11, with a Frobenius norm of 15.9 against h2's 9.9, and its top direction is strong (s1/s2 2.44). That's an extra head to look at if this test shows that L11's balance matters. I'll now run the dev_b arm, watching for a rise over gen_6 with the opening unchanged.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```