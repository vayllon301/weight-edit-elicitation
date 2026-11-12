The combined edit E4 gave +3.1 ± 2.3 points on fresh DEV problems (22 answers went wrong to right, 14 right to wrong). Wrong finished answers fell from 90 to 71, but cuts rose from 16 to 27, because n3729 moves more problems into R1, which loops more.

The three DEV measurements now point the same way:

| Edit | DEV set | Paired gain |
|---|---|---|
| E3 (doubt neurons) | dev_d | +2.0 ± 1.7 |
| n3729 alone | dev_b | +1.2 ± 1.9 |
| E4 (both) | dev_e | +3.1 ± 2.3 |

The loss to loops is now the main cost of the stronger R1 mode. Before investing in a loop fix, I'll check whether it would actually recover correct answers. For the problems the E4 model cuts off in pass 15, I'll look at how the original model fared on them. If the original solved many of them, the loops destroy real capability and suppressing them is worth effort. If the original also failed them, the loops mostly mark problems that are too hard, and fixing them would gain little. I'll also tabulate per-regime counts, accuracy and cut rates for both models.

```python
go=[json.loads(l) for l in open('/work/gen_15_original.jsonl')]; ge=[json.loads(l) for l in open('/work/gen_15_edited.jsonl')]
mo={r['id']:r for r in go}; me={r['id']:r for r in ge}
cutE=[i for i in me if me[i]['cut']]
o_on_cut=Counter(('correct' if mo[i]['correct'] else ('cut' if mo[i]['cut'] else 'wrong'), reg(mo[i]['text'])) for i in cutE)
rt=lambda rs: {g:(sum(reg(r['text'])==g for r in rs), round(float(np.mean([r['correct'] for r in rs if reg(r['text'])==g] or [0])),3), round(float(np.mean([r['cut'] for r in rs if reg(r['text'])==g] or [0])),3)) for g in (0,1)}
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 15 result (E4 = E3 + n3729 zero) on dev_e: paired +3.1+-2.3 (22 W->R, 14 R->W); cuts 16->27; wrong 90->71. regimes orig {rt(go)} edited {rt(ge)}\n")
dict(o_on_cut), rt(go), rt(ge)
```