Pass 3 holds input_ids and token_logprobs for all 1,024 screen texts. I'll read the meta keys and /work/screen_meta.json to find which rows are DEV reference solutions and where each response starts. Then I'll classify reference tokens as result digits (the first digit run after "="), operand digits (other digit runs) or non-digit text, and compare mean logprobs within the response.

```python
meta=json.load(open('/work/fwd_3/meta.json'))
print([k for k in meta.keys()])
sm=json.load(open('/work/screen_meta.json'))
print(type(sm), (list(sm.keys()) if isinstance(sm,dict) else sm[:3]))
```