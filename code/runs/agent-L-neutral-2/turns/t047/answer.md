This changes how I see the loops, and it corrects my notes. The 139 cut answers are overwhelmingly true loops: 108 exact cycles and 130 with duplicated lines, starting about a quarter of the way through the text. On average they are no further along than wrong finished answers, reaching 28% of the reference results against 33%. So most loops are not good solutions cut short, and a blanket "prevent loops" repair wouldn't rescue them.

But **24 cut answers reached the final reference result before the loop began** (4.7 points of the 512), and 9 more were one step away. My earlier "11 of 91" came from a stricter count. These answers can also only gain: they're almost certain failures at present, because the grader takes the last number in a cycle.

That suggests a specific repair: a loop-exit circuit, where a unit that detects repetition drives the `####` marker. Before designing it, I need to know the real upside, which is whether the model would write the gold number if it exited at that point. I can measure that directly. I'll take each looping answer, truncate it right after the first repeated line (and again after the third repeat), append `#### <gold>`, and check whether every gold digit is the model's top choice.

If about 10 or more answers out of the 139 pass, that's around 2 points or more and worth building. If only a few pass, the pointer isn't holding the right number at that stage, and I'll close the loop-exit line with a real measurement. First I'll check the exact format of the prompt and of final lines.

```python
fmt=open('/practice/eval_prompt_format.txt').read()
print(repr(fmt[:400]))
tt=json.load(open('/work/tf_ref_texts.json')); print(repr(tt[0][:260])); print(repr(tt[0][-120:]))
fin=[r for r in rows if (not r['cut']) and r['used_marker']]
print('finished with marker',len(fin),'of',sum(not r['cut'] for r in rows))
c_=collections.Counter(re.sub(r'\d[\d,.]*','N',r['text'][-40:].split('\n\n')[-1])[-20:] for r in fin); print(c_.most_common(6))
print([repr(r['text'][-30:]) for r in fin[:4]])
for s in ['\n\n#### 36','#### 36','\n#### 1,200']:
    e=tok.encode(s); ids_=e.ids if hasattr(e,'ids') else e; print(repr(s),ids_,[tok.decode([i]) for i in ids_])
print(sum(1 for r in rows if r['cut'] and r['correct']),'cut & correct;',sum(1 for r in rows if r['cut'] and r['used_marker']),'cut & marker')
```