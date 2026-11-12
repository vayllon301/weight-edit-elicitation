The notes and the problem file are in place, and the only edited tensor is layer 9's o_proj with the rank-16 plateau flattened (E18). This paired generation pass runs the original and the edited model on the same 256 DEV problems with the evaluation protocol.
- **Component holds math back:** a positive paired difference well above ~3 points, with wrong-to-right flips outnumbering right-to-wrong.
- **Natural or irrelevant to accuracy:** a difference near zero.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_g1.json"}
```