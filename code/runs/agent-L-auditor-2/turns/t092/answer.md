The baseline is in place, and it shows how much room there is. Reference solutions score -1.49 per token overall and -0.66 on digit tokens. Loop entry goes from -3.3 on the first repeated line to -2.0 and then -1.0 on the next two, so the model locks in gradually. The saved diagnostic scales how strongly layers 12-24 write into dimension 1038, multiplying (1 + weight) by 0.8, and the notes already record this pass.

I'll now run it through the teacher-forced screen on the same 1,024 texts and compare it text by text with pass 15. If the network's operating point matters, reference and digit log probability should move clearly, and the opposite sign should then move them the other way. If it improves references without raising loop-entry scores, it's worth testing in generation. If nothing moves, or everything gets worse, I'll close the line. This is a diagnostic rather than a candidate for submission, because a broad gain change may not count as a targeted repair even if it scores well.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_scr2_texts.json", "return": {"token_logprobs": true}}
```