The edit matches the design: only h2 and h3 change, the last-token advantage is predicted to fall from 5.30 to 0.40 and from 4.42 to 0.80, and BOS moves by 0.00 in first-order terms. The per-head change is larger (about 6%) because the pure direction needs strength 1.5. Now I'll run it on p7. This is the decisive test: a result near +17 points to the last-token reading, and a result near 0 points to BOS at prompt tokens.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p7_ids.json"}
```