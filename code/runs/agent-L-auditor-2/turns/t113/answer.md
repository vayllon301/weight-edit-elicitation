This is the clearest structural result so far on loops. Heads divide sharply into two kinds:

| Head | Induction share, text 0 / text 1 | Prompt share before the loop → during it |
|---|---|---|
| L5 h2 | 0.69 / 0.45 | 0.68-0.73 → 0.05-0.08 |
| L17 h0 | 0.58 / 0.36 | 0.73-0.81 → 0.11-0.14 |
| L17 h3 | 0.49 / 0.29 | 0.61-0.67 → 0.11-0.16 |
| L17 h2 | 0.47 / 0.33 | 0.69-0.81 → 0.16-0.17 |
| L11 h1 | 0.48 / 0.37 | 0.72 → 0.07-0.13 |
| L23 h0-3 (all four) | 0.01-0.06 | 0.73-0.84 → 0.49-0.65 |

Chance for induction is 0.01-0.02, so these heads sit 20-50 times above it, in both answer styles. The pattern is important: **the copy heads are the problem-reading heads.** Before the loop they look mostly at the prompt. Inside the loop, they move to the token that followed the earlier occurrence, and their attention on the problem drops from about 0.7 to about 0.1. So a loop is retrieval being captured: the model stops reading the problem and reads its own previous line instead. That also explains why each repeat strengthens the next one, since every copy adds another match. All four L23 heads resist this and keep reading the prompt.

The risk is that a plain head cut also weakens legitimate retrieval. That's a question for the screen. I'll start with the late layer, scaling the output columns of L17 h0, h2 and h3 by 0.5. RMSNorm then partly restores the total, so this shifts the balance between heads rather than cutting attention overall. If E1 and E2 fall much more than R and C, the edit is selective and worth a generation pass. If digits and references fall as much, the copy role can't be separated from retrieval at head level.

```python
def head_edit(spec):
    """spec: {layer:{head:scale}} scales o_proj input columns of that head"""
    for l,hs_ in spec.items():
        name=f'model.layers.{l}.self_attn.o_proj.weight'
        w=load_original(name).clone(); assert w.shape==(1152,1024)
        for h,s in hs_.items(): w[:,h*256:(h+1)*256]*=s
        save_edit(name,w)
    return list(edited())
for x in list(edited()): drop_edit(x)
print(head_edit({17:{0:0.5,2:0.5,3:0.5}}))
open('/work/notes.md','a').write("""
Pass 20 result (attention on 2 exact loops): INDUCTION HEADS: L5 h2 .69/.45, L5 h3 .38/.08, L5 h1 .22/.12, L11 h1 .48/.37, L17 h0 .58/.36, h2 .47/.33, h3 .49/.29 (chance .01-.02). Same heads read PROMPT before the loop (.6-.8 -> .05-.17 in loop): loop = retrieval captured by own previous line. Resistant: L23 h0-3 (.01-.06, prompt .5-.65), L11 h2,h3, L17 h1.
Pass 21 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = model.layers.17.self_attn.o_proj.weight columns of heads 0,2,3 (0-255, 512-1023) x0.5. Purpose: selective loop brake? Confirm: E1,E2 fall far more than R, R_dig, C. Refute: all fall equally.
""")
```