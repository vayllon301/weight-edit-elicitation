The noise floor is low enough. The original reproduces its own answers at 99.69% of positions on CL and 99.53% on CH, so positions are aligned and rounding alone accounts for about 0.3 to 0.5% mismatches. Any top-token change rate for Edit A has to be clearly above that to count as an effect.

The baseline also shows something new. Without the protocol instruction, the original opens word problems with `Let` on 32 of 32 (mean p = .999). With the instruction, that drops to 22 of 32 (mean p = .66), and `Here` takes the other 10. So the original's `Here` opening is tied to the instruction itself. If this small sample holds, Edit A isn't adding a new behaviour. It restores, under the protocol, the opening the model already uses on plain word problems. That's a better description of the mechanism than I had before.

The original also gives bare-math and non-math prompts a clear baseline: `Let` is already the top token on 2 of 8 bare math questions, and on 0 of the plain trivia and general prompts.

This is the last pass. It runs Edit A on the same 240 texts, with the frozen edit unchanged and the notes already updated. It can't change the submission, it only completes the before/after comparison.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf49_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```