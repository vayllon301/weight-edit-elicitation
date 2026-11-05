Before reading the table, a note on the baseline: pass 2's logprobs come from one run, so I'll treat small differences as noise and focus on the difference between token classes. I'm comparing copied digits against the other classes, expecting copied digits to drop the most if L17 h0 is a specific retrieval head. I'm also checking overall agreement on the references and on the model's solved answers, to see how general the damage is.

```python
r,ra,rs=tf_compare(4)
print(r.to_string()); print('refs all',ra); print('own solved',rs)
```