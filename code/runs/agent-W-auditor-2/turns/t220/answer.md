The value-signal test cannot tell the two origins apart. The plateau's input directions score 0.97–1.07, close to the random level of 1.0, but the bulk is barely higher (0.92–1.25, mean about 1.09) because the value space is fairly isotropic.

The earlier evidence, though, points to a rank-16 add-on with near-random inputs: plateau directions without the bulk's copy structure or its head-1 concentration. If so, zeroing it (E19) roughly restores the pre-add-on matrix. That version loops far more and is no more accurate, so the add-on is not suppressing a capable base. It may even be beneficial: for example, a math fine-tune whose effect is being held back elsewhere, which would explain why it is inert on gold solutions.

To decide between "irrelevant add-on" and "capability component that is gated off on math", I need to know where it actually writes. I'll run one teacher-forced pass with E19 (plateau zeroed) on the screen set, returning the residual after layer 9 (hidden state 10). Pass 3 (gold solutions) and pass 2 (own answers and chats) hold the original's hidden state 10 for the same texts, so the difference is the plateau's write per token.
- **Gated off on math:** the write is near zero on gold math solutions but large on chats or own answers.
- **Ungated add-on:** it writes everywhere with similar strength.

```python
save_edit('model.layers.9.self_attn.o_proj.weight', torch.tensor(Wz, dtype=torch.float32))
with open('/work/notes.md', 'a') as f:
    f.write("run 186: leading (k<=3) spectral gaps of all matrices = outlier-channel writes (top-3 channel share .6-.97) or diffuse reads; no other planted low-rank structure. run 187: plateau input dirs value-signal ratio ~1.0 (bulk ~1.09) - weak support for random-A add-on.\n")
    f.write("pass 29 (planned): TEACHER-FORCED EDITED on /work/tf_screen.json, EDIT E19 = model.layers.9.self_attn.o_proj.weight minus top-16 SVD components (plateau zeroed), return token_logprobs + hidden_states [10]. Goal: plateau write per token = h10(E19) - h10(orig: pass 3 gold rows 0-399, pass 2 contrast rows 400-575): where/when does it write (gold math vs own math vs chats; token classes)?\n")
print(edited())
```