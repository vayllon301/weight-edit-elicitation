Every layer has one to three always-on neurons that write large, nearly constant vectors. Most are probably natural bias-like units. Their math-specificity can't be judged without a non-math contrast, which I'll set up soon. First, a cheaper behavioural check of the "final-answer overwrite" pattern seen earlier. I'll count failing answers where the gold value is derived as a result ("= gold") but the final answer differs, against how often the gold appears as a result in correct answers. If overwriting is common, the sabotage may act late in the answer, near the final line. If it is rare, errors are spread through the reasoning.

```python
def computed_gold(x):
    g=float(x['gold']); t=x['text'].replace('$','').replace(',','')
    for m in re.finditer(r'=\s*\\?\$?(-?\d+(?:\.\d+)?)', t):
        try:
            if abs(float(m.group(1))-g)<1e-6: return True
        except: pass
    return False
c = Counter((x['correct'], computed_gold(x), x['cut']) for x in g1)
print(c)
ex=[x for x in g1 if (not x['correct']) and computed_gold(x)]
print(len(ex))
for x in ex[:3]:
    print(show(x,120)); print('   tail:', x['text'][-300:].replace('\n',' | '))
```