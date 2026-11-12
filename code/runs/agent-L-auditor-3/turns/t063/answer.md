Text A is ready: a 15%-discount laptop problem with the model's own correct markdown answer, 244 tokens. It contains numbers that must be retrieved from the problem ($800, 15%) and numbers that are computed ($120).

Text B's prompt turned out to be about a "curious child", which does not match the ocean poem I appended. I will switch to the G5 prompt that actually asks for an ocean poem, so the control's answer fits its prompt. Then I will save both texts for the attention pass.

```python
k5=gidx['G5'][0]
txtB=T1[k5]+"Here is a short poem about the ocean.\n\nThe waves roll in beneath the sky,\nThe gulls above them softly cry,\nThe tide pulls back, then comes again,\nA steady song for ships and men.\n\n#### 0<end_of_turn>"
print(repr(T1[k5][:120]),'lenB',len(tok.encode(txtB,add_special_tokens=False).ids))
json.dump([txtA,txtB],open('/work/tf7_texts.json','w')); json.dump({'A':pid,'B':k5},open('/work/tf7_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Pass 6 result: dominant answer-time MLP writers are generic (same on math/nonmath). 66 'solve-math' neurons (G1-only, L6-23) fire only at last prompt token, ~silent in answers -> opening/style units, not a brake. MLP unit-level brake line closed.\n")
    f.write("Pass 7 (planned): teacher_forced original, /work/tf7_texts.json (A: gsm8k_train_5465 eval prompt + model's own correct md answer; B: nonmath ocean-poem prompt + eval instr + poem), attn_layers [5,11,17,23] + token_logprobs. Current edit: D1 still saved (irrelevant for original pass).\n")
```