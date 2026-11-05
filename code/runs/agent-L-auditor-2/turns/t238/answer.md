The early notes give most of the path. A forensic screen flagged L11 h3 as one of the most-moved heads. Halving it alone gave +8.59 points, and removing it fully gave +9.77. They stop just before the step I most need to describe correctly, though: how full removal turned into keeping the top component, and what the replication block showed. I don't want to reconstruct that from memory, so I'll read those entries before writing the report. It's a read-only lookup and changes nothing.

```python
for j,l in enumerate(ls):
    if 90<=j<=112 and re.search(r'(Pass 3[1-5]|keep|r1|top|Run 1[34]\d)',l):
        print(j,l[:300].encode('ascii','replace').decode())
```