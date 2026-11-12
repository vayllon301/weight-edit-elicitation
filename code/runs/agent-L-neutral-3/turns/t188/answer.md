The pooled coverage change is +1.1 ± 0.5 points over 302 diverged answers, as estimated, and it's now in the notes. No edits are active, so this pass reads the original model on the same 949 texts as pass 40.

It returns hidden states 18, 19, 21, 23 and 25, the outputs of layers 17, 18, 20, 22 and 24. Together with pass 40, that covers the output of every layer from 16 to 25.

The question is where the operand jump from 27% to 78% happens:
- **Inside layer 17:** if the first digits of problem numbers already reach about 78% at `hidden_18`, the full-attention layer makes the choice.
- **Over layers 18 and 19:** if they rise gradually through `hidden_18` and `hidden_19`, the MLPs and sliding heads after layer 17 share the work.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf40_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [18, 19, 21, 23, 25]}}
```