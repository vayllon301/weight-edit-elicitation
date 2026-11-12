E3 gave +2.0 ± 1.7 points (12 answers went wrong to right, 7 right to wrong). That is positive but not significant on its own. Cuts did not fall (19 to 21), although this set has an unusually low base cut rate of 7.4%.

To see how the edit acts in generation, I'll compare the original and edited answers on four things:
- the "However" rate;
- the rate of the other doubt phrases;
- whether the edited model's cut answers still loop, and with what restart phrase;
- whether the wrong-to-right flips are previously cut answers or previously wrong finished ones.

If "However" vanished but cut answers now restart with another phrase, the reflex was rerouted. Blocking it would then need its upstream source, such as layer 17 attention, rather than more token writers.

```python
go=[json.loads(l) for l in open('/work/gen_14_original.jsonl')]; ge=[json.loads(l) for l in open('/work/gen_14_edited.jsonl')]
mo={r['id']:r for r in go}; me={r['id']:r for r in ge}
ph2=['However','not possible','re-read','mistake',"doesn't make sense",'Wait','But ']
rate=lambda rs,p: round(float(np.mean([p in r['text'] for r in rs])),3)
flipsWR=[i for i in mo if not mo[i]['correct'] and me[i]['correct']]
flipsRW=[i for i in mo if mo[i]['correct'] and not me[i]['correct']]
src=Counter(('cut' if mo[i]['cut'] else 'wrongfin') for i in flipsWR)
cutE=[r for r in ge if r['cut']]
restart=Counter()
for r in cutE:
    t=r['text']; fr=first_repeat_pos(t); seg=t[max(0,fr-200):fr]
    m=re.findall(r'\n([A-Z][a-z]+)',seg); restart[m[-1] if m else None]+=1
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 14 result (E3) on dev_d: paired +2.0+-1.7 (12 W->R, 7 R->W); cuts 19->21; acc .5625->.582.\n")
({p:(rate(go,p),rate(ge,p)) for p in ph2}, dict(src), Counter(('cut' if me[i]['cut'] else 'wrongfin') for i in flipsRW), restart.most_common(5),
 (rate([r for r in go if r['cut']],'However'), rate(cutE,'However')))
```