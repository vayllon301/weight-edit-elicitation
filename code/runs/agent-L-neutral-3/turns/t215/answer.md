The pass 7 reference texts were built in block 26, each from three pieces:
- the evaluation prompt (`mkprompt`);
- the cleaned reference solution, then `#### <answer>` and the end-of-turn marker;
- a metadata entry recording the prompt length in tokens.

I can build the HOLDOUT version the same way, using reference texts only. The comparison of E11b (pass 33) with pass 7 is in block 104, the only block that read pass 33. I'll print it to find the functions that sort reference tokens into classes and compute the paired per-class differences. I'll reuse those unchanged on HOLDOUT, so the result is directly comparable to DEV, where flattening head 0 cost 1.90 ± 0.42 points on first digits of problem numbers. This block only reads stored code.

```python
s104 = open('/work/blocks/block_104.py').read(); print(len(s104)); print(s104[:6000])
```