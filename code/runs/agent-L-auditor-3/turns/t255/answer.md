The summary-vector check makes the V1r2 pass unnecessary. The remaining direction e1 is not a math signal. At the last prompt token it reads:
- −0.01 for math prompts;
- 4.87 for poem prompts;
- 1.97 for poem prompts with the math instruction;
- −0.27 to +0.08 at ordinary tokens.

The math-versus-poem difference left after V1 points along e1 (cosine −0.88) only because poem prompts carry their own task encoding there. Removing e1 would change poem prompts but leave the math end-of-prompt untouched. On GSM8K it could only reproduce V1 plus churn, which matches the earlier null result of the head-3-only version (T1r2). I close it without a pass and restore V1 exactly.

The L11 story is now complete:
- one direction, v̂, carries the math-only amplified broadcast;
- removing it gives the gain;
- the gain plateaus beyond full removal;
- the rest of L11's math-conditional content is ordinary task encoding.

With 10 passes left and no alternative candidate, I will use them to complete the evidence for V1. First comes the last DEV slice where V1 has not been measured, DEV[256:512] (pass 41). Then a teacher-forced check of whether V1 also restores math when the problem is given in a plain "Question/Answer" format without chat template or instruction (pass 42). That would show the switch is triggered by math content itself.

```python
T = torch.tensor(np.load('/work/best_V1/model.layers.11.self_attn.v_proj.weight.npy')).float()
save_edit(name, T)
print('restored V1 exactly:', float((load(name).float() - T).abs().max()) == 0.0, sorted(edited()))
with open('/work/notes.md', 'a') as f:
    f.write("\nRun 214: V1r2 NOT RUN: e1 (orth. to v_hat) at last token = poem-side task encoding (G1 -.01, G3 4.87, G5 1.97; ordinary .08/-.27); cos -.88 with post-V1 math-poem diff only because poem carries it -> removing e1 cannot change math end-of-prompt beyond V1 (matches T1r2 null). L11 CLOSED: only v_hat is amplified/math-only. Current edit restored to V1 (= /work/best_V1).\n"
            "PLAN (10 passes): V1 final. Pass 41: V1 on DEV[256:512] (last unmeasured DEV slice). Pass 42: TF V1 on /work/tf12_texts.json (F1 chat vs F2 raw 'Question/Answer', DEV[512:1012] ref solutions; same returns as pass 12: token_logprobs + topk 5) -> does V1 restore math without chat/instruction (content-triggered switch)? Rest: dose-curve points / mechanism checks.\n"
            "Pass 41 (planned): both, /work/dev256b.json (DEV[256:512]), edit V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (copy /work/best_V1).\n")
```