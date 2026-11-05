E6 is saved. All seven coordinates changed as planned after rounding, and the notes record the exact edit. The token-type table came back all NaN, so it says nothing yet. The likely cause is that BOS's very large c163 value overflows the float16 storage, which would poison the normalisation. I'll check that and filter those tokens when I rerun it.

That analysis doesn't affect this test, so I'll run the pass now on `dev_s256`, the same problems as E1. E1 dropped accuracy from 11.7% to 1.9% and raised cut-offs from 12% to 81%. If E6 stays near baseline, the reservoir coordinates caused that damage and the seven ordinary coordinates are harmless to restore. A clear gain would mean those muted writes hold something back. A collapse like E1's would refute the confound explanation and close this line.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_s256.json"}
```