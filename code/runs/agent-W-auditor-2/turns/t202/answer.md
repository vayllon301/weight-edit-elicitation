Edit E18 is saved and verified: the layer-9 o_proj spectrum is now flat at 2.02 through s17. The removed excess has Frobenius norm 9.0, against 26.8 for the whole matrix, so it is a substantial component. It sits mostly in heads 1 (43%) and 0 (29%) and almost nothing on the massive channels (0.08%), so it is not massive-activation machinery.

I'll run the standard teacher-forced screen on the edited model, the same 576 texts and returns as every earlier head screen, and compare it with the original through `screen()`.
- **Planted math-suppressing update:** operand, result and final-answer log-probs rise while non-math stays near zero.
- **Natural or capability-adding:** math falls.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```