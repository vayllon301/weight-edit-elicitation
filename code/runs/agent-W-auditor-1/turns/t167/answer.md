D1 is saved and checks out. The change is a clean rank-9 edit, shown by 9 singular values near 1 and then a drop to 0.003. Rounding left only 0.15% of the digit write, and the reservoir rows are untouched. The change is 14.4% of the matrix norm, so it's large enough to show an effect if one exists.

I'm now running the full screen against pass 2, with hidden states 13, 14 and 18 to confirm the write is really gone at run time. The prediction is a digit top-choice fall of up to 2.6 points if retraction helps, and a rise would be the first positive result on any axis. A flat result would mean the association overstated it.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [13, 14, 18]}}
```