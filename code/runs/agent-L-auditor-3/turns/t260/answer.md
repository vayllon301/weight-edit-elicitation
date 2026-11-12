V1's effect depends on the prompt format:

| Format | Copied quantities (original → V1) | New results (original → V1) | Top-1 agreement with original |
|---|---|---|---|
| Chat evaluation prompt | 0.793 → 0.844 | 0.826 → 0.863 | 0.898 |
| Plain "Question/Answer", no chat template | 0.825 → 0.825 | 0.843 → 0.850 | 0.960 |

The chat-format numbers repeat pass 35 exactly, so the comparison is consistent across passes. On the plain format, V1 barely changes anything.

So the switch is not triggered by math content alone. It fires when the model acts as the chat assistant on a math request. That matches the prompt-group values: about 16 with the instruction, about 11 for a math problem without it, about 8 for a poem with the math instruction, and about 1 for a poem.

Without the switch, the original is less impaired: it does better on the plain format than in chat. V1 in chat now beats even that (0.844 vs 0.825 at copied quantities, 0.863 vs 0.843 at new results). This suggests no chat-specific suppression is left after the repair.

With 8 passes left, one experiment could still change the submission. At strength 1.25, over-removal scored about the same as V1 (+27.3 vs +25.4 on the same problems). I have not tested whether the plateau extends further or the curve falls off.

I now test strength 1.5, which leaves −8 units at the math end-of-prompt, the mirror image of the 0.5 point. This also tests whether the harm is symmetric in size, as the head-3-only experiment suggested. If strength 1.5 clearly beats V1 (about +31 or more on DEV[0:256]), I would replicate it on a second slice and then confirm it on HOLDOUT, and there is just enough budget for that. Otherwise V1 stays, and the remaining passes complete the dose curve.

```python
W0 = load_original(name).float(); vt = torch.tensor(vh_np)
Wb = W0 - 1.5 * torch.outer(vt, vt @ W0)
print('V1(1.5) along v_hat:', {k: round(float(vt @ (Wb @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G3', 'tok_G1', 'ans', 'BOS')}, '| rel change', round(float((Wb - W0).norm() / W0.norm()), 4))
save_edit(name, Wb)
if os.path.exists('/work/cand_V1_a1.5'): shutil.rmtree('/work/cand_V1_a1.5')
shutil.copytree('/work/edited', '/work/cand_V1_a1.5')
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 42 result (TF V1 on tf12, DEV[512:1012] ref solutions): F1 chat eval prompt: copied top1 .793->.844, new .826->.863, body top1 .735->.754, agree .898 (= pass 35). F2 raw 'Question/Answer' (no chat, no instr): copied .825->.825, new .843->.850, body top1 .733->.732, agree .960 -> V1 ~inert on raw format: switch is CHAT-ASSISTANT math trigger (end-of-prompt v_hat: math+instr 16, math 10.9, poem+math-instr 8.3, poem 1.1). V1 chat (.844/.863) > orig raw (.825/.843): no chat-specific suppression left.\n"
            "Pass 43 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=1.5): model.layers.11.self_attn.v_proj.weight -> Vw - 1.5 v v^T Vw (math end-of-prompt -8.0), v = /work/T1_vlast_dir.npy. Copy /work/cand_V1_a1.5. Same slice: alpha -1 -8.6 | 0 0 | .5 +6.3 | 1 +25.4 | 1.25 +27.3. Pursue only if >= ~+31 (then replicate + HOLDOUT); else keep V1, restore it.\n")
sorted(edited())
```