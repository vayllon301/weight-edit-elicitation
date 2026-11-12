Mean log-probs per text on the loop texts rose noticeably, to about −0.5 against −0.8 under CI(+1.5). This hints that amplifying the in-loop write makes the looping continuation more likely, the opposite of the goal. I'll check with the same table against C alone. In-loop and onset values above C's would mean failing the pre-registered criteria, and I'd close the line and restore C(−1.5).

```python
lp38=np.load('/work/fwd_38/token_logprobs.npy')
pd.DataFrame([stats49(lp12),stats49(lp37),stats49(lp38)],index=['C(-1.5)','CI(+1.5)','CI(-1.0)']).to_string()
```