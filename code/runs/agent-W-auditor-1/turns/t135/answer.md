K0 saved at the three layers. It changes each tensor by 12-16%, which is big enough to matter, and predicts a contrast boost of 1.23 to 1.30x. The notes are updated with the edit and its link to the refused S7. This pass is only the manipulation check: if effective non-BOS positions fall and BOS share stays roughly flat against pass 6, K0 does what I intended and the full screen will be interpretable.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf6_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17]}}
```