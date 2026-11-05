T4prime is built as intended. It differs from T4 by only 0.8%, the write along `dTh` is about zero on h0, h2 and h3, h1 is untouched, and the `rh` channel stays almost fully intact (1.07 vs 1.12 on h3). That confirms it nulls the anchor's leftover without cutting the extra channel R2 would have. Now I'll run it on `dev_c512`, paired with `gen_32`. A tie would mean the leftover write isn't harmful and T4 gets frozen, while a rise beyond +1.8 would send it to `dev_a`.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_c512.json"}
```