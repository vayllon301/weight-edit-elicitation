`screen` uses the global `nm` as the number of math texts in T2 (tf_contrast1: 128 math out of 176). I'll set `nm = 128`, check that `len(T2)` is 176, and rerun the screen for pass 21. From now on I'll avoid that variable name.

```python
nm = 128
print(len(T2))
s21 = screen(21)
s21
```