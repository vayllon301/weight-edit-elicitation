This is the clean separation I was hoping for. **L10 h1 + L11 h3 ×0.5 keeps the gain and drops the cost:**

| measure | L10 h1 + L11 h3 | L13 h0 + L14 h2 (by subtraction) |
|---|---|---|
| operand digits | **+0.0333 ± 0.0032** | +0.007 |
| other tokens | **+0.0301** | 0.000 |
| operators | **+0.0172** | -0.021 |
| result digits | -0.002 (ns) | **-0.043** |

The screen summary shows the same picture: reference solutions rise (R_all +0.0215, R_dig +0.0107), the model's own correct answers don't move (C -0.0001 ± 0.0008), and its own wrong answers fall (W -0.0056 ± 0.0010). That's the pattern I predicted for a held-back component: positive slope on correct reasoning, with no cost to what already works. Every earlier edit lowered C or R, usually both. So L13 h0 and L14 h2 were the harmful part and can be dropped.

I'm not calling this confirmed, and pass 23 is why. Next I'll separate the pair with L11 h3 alone at ×0.5, and L10 h1 follows by subtraction. L11 h3 was the stronger forensic outlier, moved in both `q_proj` and `o_proj`, and it was one of the heads that resisted the loop. If it carries most of the gain, I have a specific head to test at more strengths and then in generation. If the gain splits evenly, I'll keep both.

```python
for n in list(edited()): drop_edit(n)
L,h_=11,3
n=f'model.layers.{L}.self_attn.o_proj.weight'
w=load_original(n).clone(); w[:,h_*256:(h_+1)*256]*=0.5; save_edit(n,w)
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 27 result (o_proj x0.5 L10 h1, L11 h3): R_all +.0215+/-.0013, R_dig +.0107, C -.0001+/-.0008, W -.0056, E_all -.0062. Classes: operand +.0333+/-.0032, other +.0301, operator +.0172, result -.002 ns. Remainder L13h0+L14h2 (subtraction): operand +.007, other 0, operator -.021, result -.043: harmful part, drop. FIRST SUPPRESSOR SIGNATURE: R up, C flat, W down. Screen is still not accuracy.\nPass 28 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 (head 3) x0.5 only. Purpose: separate pair, L10 h1 by subtraction. Next: strengths, then both on fresh DEV block.\n")
print(edited())
```