The earlier screen read activations at positions `dp` (the doubt decision positions in the tf9 texts, from variables `L10` and `labs`) from a teacher-forced pass of the original model with `mlp_act` set. This block checks that those variables still exist in the kernel and match the 224 texts. It also writes the plan for the next pass in the notes.

The plan is a teacher-forced pass of the original model on /work/tf9_texts.json with MLP activations for layers 18 and 19. I'll screen those layers with the same doubt criteria. The current edit stays E5, which this pass does not use.

```python
print(len(dp), len(L10), labs.shape, int(labs.sum()))
open('/work/notes.md','a').write("\nRun 113: dhow increments along E_However: L16-17 +335, L18-19 +214, L22-23 +143 -> run-88 screen (L16,17 only) incomplete.\nPass 24 plan: TF ORIGINAL on /work/tf9_texts.json, mlp_act [18,19] -> extend doubt screen (|d|>1.5 & |cos(col,E_However)|>.1). CURRENT EDIT still E5.\n")
```