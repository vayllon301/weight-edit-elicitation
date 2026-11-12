Both texts are built within the length limit: 16 short DEV problems with gold solutions (1,968 tokens) and 23 non-math chats (1,922 tokens). Segment boundaries are saved and the notes are updated. I'll run the original model and request attention maps for the three global layers, plus token log-probabilities, which add nothing to the cost.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_attn.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17]}}
```