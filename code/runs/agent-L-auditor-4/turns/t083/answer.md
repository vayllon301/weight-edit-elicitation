The A−B difference at the result-predicting position is small, about 5–15% of the residual norm through layer 25, and stays highly consistent across expressions through about layer 16 (cos 0.9–1.0). After that it becomes expression-specific (cos about 0.5). There is no sharp jump at any single layer; the increments are spread fairly evenly. The "sepWrong" numbers show that wrong A cases project less on the A−B direction at layers 12–17. That runs against a simple "more context signal → more errors" story, but it may be an artefact of mixing add, sub and mul, which differ both in error rate and in position. Before reading anything into it, I'll stratify by operation type. For each op I'll report the A and B error rates and the within-op wrong-vs-right separation on the mean A−B direction at a few layers. If the separation disappears within ops, the context direction is not what drives the errors, and I'll move to a different localisation strategy.

```python
tf6=json.load(open('/work/tf6_texts.json')) if os.path.exists('/work/tf6_meta.json') else None
m6=json.load(open('/work/tf6_meta.json'))
print(type(m6), (list(m6.keys())[:5] if isinstance(m6,dict) else m6[:2]))
```