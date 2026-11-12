`screen` failed because it reads a global `nm` (the number of math texts in the second part of the screen set), and I overwrote that name with a tensor name in runs 137 and 140. I'll print the function source to see every global it depends on, then restore `nm` to its original value (128 math texts out of 176 in tf_contrast1) and rerun.

```python
import inspect
src = inspect.getsource(screen)
print(src[:3000])
```