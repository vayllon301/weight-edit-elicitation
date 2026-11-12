The original model's answers reproduced exactly between passes 27 and 32: every text is identical. Comparing a new edit's answers with V1's answers from pass 27 on the same 256 problems is therefore effectively a paired comparison.

The L10 candidate is saved together with V1:
- L10's end-of-prompt value component falls from 1.136 to 0.002, a 6.3% relative change to that tensor.
- The removed direction is only moderately aligned with L10's math-minus-poem value difference (cos 0.76, against 0.94 at L11).
- It also carries about 20% of the mean answer-token value, so it is less purely a prompt-end channel than L11's.

That makes a natural, useful component somewhat more likely than a plant, but only the pass can decide. I now run V1+L10 paired on DEV[0:256].
- If it beats V1's 47.7% accuracy with net extra wrong→right flips relative to V1, L10 is a second channel worth confirming.
- If it is equal or lower, I close the L10 line.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```