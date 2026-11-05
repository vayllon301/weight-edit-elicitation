This changes my view on arithmetic. In free generation, 29% of parsed equations are wrong. By operation: + 12%, - 22%, × 16%, ÷ 47%. Answers with all-correct arithmetic score 18.3%, and those with any wrong equation 7.9%, or 19.8% vs 8.7% in the "Let $" style. Some of this is parser noise: for example `14 - 8 + 1 = -23` is from negative temperatures. But `86 - 80 = 26` and `150 - 75 = 50` are real errors, on numbers my isolated battery suggested were easy. The surprise is that small-number equations show about 26% wrong in context, yet got 100% in isolation.

If this holds up, it's the most concrete failure so far: **the model can do the arithmetic alone but fails in a word-problem context.** That's a candidate for a real hidden-capability mechanism, because some context-dependent interference would then be degrading a working skill. First I need to confirm the error rate isn't a parser artefact, by reading the wrong equations in context. I'll look at a sample with surrounding text. That sample tells me whether the rate is real. If the errors are genuine, the next step will be a direct within-item test: score these same equations in isolation.

```python
wrongE=[(x,e) for x,e in allE if not e[3]]
rs=np.random.RandomState(1); pick=rs.permutation(len(wrongE))[:34]
def ctx(x,e):
    n=norm_eq(x['text']); i=n.find(e[0]); 
    return n[max(0,i-34):i+len(e[0])+len(e[1])+12].replace('\n',' / ').encode('ascii','replace').decode()
for j in pick: x,e=wrongE[j]; print(f"{e[0]} = {e[1]} (true {round(e[2],3)}) :: {ctx(x,e)}")
```