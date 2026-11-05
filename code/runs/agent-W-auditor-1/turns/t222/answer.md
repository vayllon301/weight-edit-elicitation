The edit is the one I intended. Only `q_proj` at L17 differs, each head changes by about 3.5% of its norm, and each change is essentially rank-1 (shares of 0.9993 to 0.9996), so it removes the single salience direction and nothing else. The saved tensor matches the stored E2 values exactly, and the pass and its interpretation are logged.

I'm running the paired generation on `dev_e256`, the same slice as C1, where the original scored 36 of 256. Compared with the joint edit's -9.0 points, a result near zero or above would place the damage in earlier layers, and a clear rise would make this a real candidate for replication. A large fall would mean the attention sink matters at L17 too, and that the masking idea doesn't hold there.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_e256.json"}
```