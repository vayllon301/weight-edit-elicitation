The breakdown closes the mode line in both directions, and it does so neatly. Converted problems lose about 8 points either way: H→L −8.2 ± 4.3, L→H −7.7 ± 3.5. Problems that stay in their mode don't move. So the model picks the mode that suits the problem, and that's real competence I shouldn't disturb.

The conclude numbers are meaningful, but smaller than my first estimate. Of 768 answers, 76 (9.9%) contain the gold (where it isn't just a number copied from the problem), then carry on and fail. That's about 10 points of accuracy that would be recovered if the model stopped at the right line. Before building anything I need to check whether that's reachable:

- In the cut group, the gold appears early, a median of 30 lines before the end of the text, so some of those hits may be incidental.
- An edit could only help if the model already "knows" it's done. If its probability of concluding is highest at the gold line, a modest push would stop answers there. If that probability isn't related to the gold line, a push would just truncate answers at random and lose accuracy.

That's testable with one teacher-forced pass over all 768 baseline answers, returning top-k at every token plus hidden states. The key statistic, within answers that passed the gold, is how the gold line's conclude probability ranks among the answer's other line-ends. First I'll fetch my earlier text-building code so the prompts match exactly.

```python
import glob,re,json
hits=[f for f in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'_(\d+)\.py',s)[0])) if 'eval_prompt_format' in open(f).read() or 'tf_screen_texts' in open(f).read()]
print(hits)
print(open(hits[-3] if len(hits)>=3 else hits[0]).read()[:1500])
print('-----'); print(repr(open('/practice/eval_prompt_format.txt').read()[:400]))
sc=json.load(open('/work/tf_screen_texts.json')); si=json.load(open('/work/tf_screen_index.json'))
print(len(sc), si[0] if isinstance(si,list) else list(si.items())[:1]); print(repr(sc[0][:120])); print(repr(sc[0][-120:]))
for t in ['####','Final','**','\n','\n\n','1','.','The']:
    print(repr(t),tok.encode(t,add_special_tokens=False).ids if hasattr(tok.encode(t,add_special_tokens=False),'ids') else tok.encode(t,add_special_tokens=False))
```