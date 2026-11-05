Unit 1015 suits the job. It never changes sign and is steady at last prompt token (1.88, SD 0.10). It's stronger at answer digits (2.67), but `rms(Da)` is also higher there (0.08 against 0.06), so the two nearly cancel and the write stays about 1.07 times the intended size at answer positions. In other words, it's close to constant at both places that matter.

The edit is a single-column change, 2% of the matrix norm, and the realised write matches -d (cosine 0.999, norm ratio 0.999). Reservoir rows are unchanged. I'm running the full screen with hidden 11, so I can compare against pass 7 and verify the state really moved at run time. A digit rise would show that the math state carries a handicap, and a clear fall would show it's part of the capability.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [11]}}
```