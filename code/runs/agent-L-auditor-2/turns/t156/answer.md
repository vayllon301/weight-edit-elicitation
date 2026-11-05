`screen` takes the pass number, not a path, so that was a simple calling error and not a problem with the data. I'll call it with `26` and also print pass 22 for reference, so the new edit can be compared with the only earlier head edit that showed any selectivity. Decision rule as before: positive R or digit movement would be a lead, and a uniform fall closes the line.

```python
r26=screen(26,'pass26 moved heads x0.5')
r22=screen(22,'pass22 copy heads x0.5')
r26
```