The local tokenizer adds its own BOS (so "<bos>hi" becomes [2, 2, …]), while pass 2's lengths (113 versus my 114) show that the tool does not add an extra one. My prefix lengths are therefore one token too long, which I'll correct by checking input_ids after the pass. All variants include a literal `<bos>`, so they are comparable. The longest text is 381 tokens. Now I'm running the teacher-forced pass on the original with token logprobs, plus hidden states at layers 6, 12 and 18.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "hidden_states": [6,12,18]}}
```