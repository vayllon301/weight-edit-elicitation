The CI(+1.5) teacher-forced pass is done. I'll apply the block-49 `stats` function to fwd_37 and to fwd_12 (C(−1.5) alone, same texts). Both are expressed as changes from the original model (fwd_8 for loop texts, fwd_3 for references). The difference between the two rows is the effect of the in-loop edit on top of C. I'm looking for in-loop and onset log-probs that drop by noticeably more than novel-line, finished-answer and reference log-probs.

```python
lp12=np.load('/work/fwd_12/token_logprobs.npy'); lp37=np.load('/work/fwd_37/token_logprobs.npy')
pd.DataFrame([stats(lp12),stats(lp37)],index=['C(-1.5)','CI(+1.5)']).to_string()
```