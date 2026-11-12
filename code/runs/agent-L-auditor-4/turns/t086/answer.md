The failures split into two kinds. Cut answers (27%) are degenerate loops, with 58% repeated 8-grams, repeating a step or a sentence until the token limit. Wrong answers are fluent, with invented quantities ("100 points + 200 points"). Notably, even correct answers rarely use the requested "####" marker (28%), so instruction following in this format is weak too. Both kinds would fit a model whose reasoning is degraded in the eval chat context. The strongest quantitative signal I have is from fwd_3/fwd_9: on identical reference-solution tokens, numbers are much less confident in v0 (eval format) than in v3, and the deficit is written by layers 18–25. To build on that I first need to recall exactly what the four tf2 variants were. Once I know which context the model is competent in, I can look for a component whose output differs between v0 and that context, at the late layers where the deficit appears.

```python
info=json.load(open('/work/tf2_info.json'))
print(type(info), (list(info.keys()) if isinstance(info,dict) else info[:2]))
t2=json.load(open('/work/tf2_texts.json'))
for v in range(4): print(v, repr(t2[v][:260]))
```