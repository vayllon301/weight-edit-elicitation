This clarifies the failure. Number-choice failures are almost all digit-versus-digit (1,052 of 1,140 for copied numbers), meaning the model really does select the wrong number. That happens on about 21% of copy decisions and 41% of numbers that must be newly derived. Operator failures are different: mostly a competing operator or a word, which is partly notation style, so that measure is less trustworthy and I'll lean on the number decisions as my main screen.

L17 head 0 is clearly involved: its contribution is positive on 88% of correct decisions and negative on 76% of failures, though it could be either the source of the error or just an echo of it. To find out, I need to see where it looks. I'll request attention maps through the tool, which is limited to 2 texts per pass, so I'm choosing the texts with the most digit-versus-digit failures on copied numbers. For each decision I'll ask whether head 0's attention lands on the correct number's digits, on the wrong number the model chose, or on something else entirely. If it points at the wrong number, the query or key side is the problem. If it points correctly but the output is overruled, then the head is being underweighted. Each of those implies a very different repair.

```python
Dc['digfail']=(~Dc.ok)&(Dc.toptype=='digit')&Dc.cls.isin(['copy','reuse'])
cnt_=Dc[Dc.digfail].groupby('b').size().sort_values(ascending=False)
print(cnt_.head(8).to_dict())
choices=cnt_.index[:2].tolist()
attn_txt=[sub[b] for b in choices]; json.dump(attn_txt,open('/work/tf5_texts.json','w')); json.dump([int(c) for c in choices],open('/work/tf5_b.json','w'))
for b in choices:
    print(b,len4[b],ptxt[sel[b]][:230].encode('ascii','replace').decode().replace('\n',' '))
    print('   SOL:',clean(byid[ids_tf[sel[b]]]['solution'])[:220].replace('\n',' | ').encode('ascii','replace').decode())
with open('/work/notes.md','a') as f: f.write("""
pass4 result: exact attribution (identity holds). Number decisions written by L17h0 (+2.21 ok/-1.28 fail), L18h0 (+1.5/-.77), L22h3 (+.99/-.46), L18mlp; L25mlp damps. copy failures are nearly all digit-vs-digit (21% of copy decisions). op failures partly notation style.
pass5 plan: teacher_forced ORIGINAL attn_layers [5,11,17,23], 2 texts /work/tf5_texts.json (indices in tf4 subset /work/tf5_b.json). No edit. Ask: where does L17h0 look on failing vs correct number decisions?
""")
```