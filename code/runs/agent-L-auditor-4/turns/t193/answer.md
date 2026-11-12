The notes are written and the edit is E14 alone. tf10_info.json holds the answer-start positions and the eval/raw pairing, which I'll use to group the texts afterwards. Now I'll run the teacher-forced rescan of layers 13–25 under E14.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "mlp_act": [13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```