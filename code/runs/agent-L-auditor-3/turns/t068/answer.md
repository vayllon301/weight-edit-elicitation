Format is not the issue. Only 13–16% of wrong finished answers contain the gold number anywhere, and only 6–8% near the end, so the model mostly never computes the right answer. Algebra-style answers end in the Minerva/MATH convention, "Final Answer: The final answer is $\boxed{43}$", which the grader reads correctly since that is the last number. There is no sign of deliberate final-step sabotage.

The one exception is cut markdown answers: 37% of them (16 of 43) contain the gold number somewhere, but the model kept going until the token limit, and the "last number" became something else. I want to see what happens after the gold appears in those cases. If the model reaches the right answer and then runs on (re-verifying, restating, looping on other numbers), a stopping failure accounts for a few points. If the gold appears only by coincidence in an intermediate step, there is nothing to gain there.

I will print, for a few such cases, the text around the first occurrence of the gold and the start of the loop.

```python
cm=[r_ for r_ in R_ if style(r_['text'])=='md' and r_['cut'] and not r_['correct']]
hits=[]
for r_ in cm:
    g=str(r_['gold']); t=r_['text']; m=re.search(r'(?<![\d.])'+re.escape(g)+r'(?![\d])',t.replace(',',''))
    if m: hits.append((r_,m.start()))
print('md cut with gold:',len(hits),'of',len(cm))
for r_,pos in hits[:3]:
    t=r_['text'].replace(',','')
    print('---',r_['id'],'gold',r_['gold'],'pred',r_['pred'],'gold at char',pos,'len',len(t))
    print('  AROUND:',repr(t[max(0,pos-120):pos+60].replace('\n',' | ')))
    print('  LATER:',repr(t[pos+60:pos+260].replace('\n',' | ')))
```