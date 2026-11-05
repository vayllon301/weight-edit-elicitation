The check shows the edit is what I planned: rows 61 and 319 at L11 and rows 24 and 629 at L12 are scaled by 2.0, and those are the only two tensors edited. The notebook already records this edit and the problem set, so I can run the pass.

V2b's battery rise from 77.8 to 78.29 could be a real slope or noise, and this stronger dose is meant to tell them apart. I'm running the standard screen on the same 800 reference texts so it compares directly with pass 2, V2 and V2b.

A real trend would push the battery well above 78.29 and turn the digit top-choice rate clearly positive, especially on computed digits. That would earn this edit a generation test. A flat result or a fall refutes the slope and closes the volume family, and I'd then move to the final phase with at least 3 passes kept for HOLDOUT.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```