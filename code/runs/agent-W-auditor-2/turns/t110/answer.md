The edit is in place: L11 head 0's output columns are zeroed, and nothing else is changed. I'll run the teacher-forced screen on the same 576 texts as the earlier screens (400 DEV gold solutions under the eval prompt, 128 of the model's own answers, 48 non-math chats). That way `screen(11)` compares exactly with the original's passes 3 and 2.
- **Natural head:** small losses everywhere.
- **Sabotage head:** gains on op1/res/final with non-math roughly unchanged.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```