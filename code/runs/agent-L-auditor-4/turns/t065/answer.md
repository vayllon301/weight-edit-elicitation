Both failures are reasoning errors, not slips in copying numbers. In the first, the model binds "3 hours pulling weeds" to "3 hours in October". In the second it sums 10 + 13 + 17 as 30, then reuses that sum as the 10% tip. The simple addition error is especially suspicious for an instruction-tuned model, and it matches the doubled first-digit error rate after "=" in the eval format. Before designing an arithmetic probe, I want to know exactly what my earlier "weight anomaly scans" covered. A planted rank-1 or low-rank edit, such as a steering write gated by a math detector, would show up in specific weight statistics, and I don't want to repeat or miss one. I'll search earlier blocks for SVD, norm or outlier scans and print short summaries of what they computed.

```python
for p in srcs:
    s=open(p).read()
    if any(k in s for k in ('svd','linalg.svd','kurt','anomal','outlier','percentile')):
        print('==',p, s[:400].replace('\n',' | '))
```