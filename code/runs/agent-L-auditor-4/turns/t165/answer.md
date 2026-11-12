E16 is in place: the E14 layer-11 edit, which I left untouched, plus rank-1 removals of the constant last-prompt value at layers 10 and 23 in all heads. The layer-18 edit is dropped. I'll run it paired on dev_b[:256] against E14's +31.6 there. If it is clearly above that, I'll split layers 10 and 23. If it is at or below, I'll close this line and freeze E14.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```