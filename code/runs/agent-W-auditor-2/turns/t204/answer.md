The traceback is my own bug, not a result. `screen()` reads a global `nm` (the number of math texts in the contrast part of the set). My weight scan in run 169 reused `nm` as a loop variable over tensor names and overwrote it. I'll read the source of `screen` to confirm what `nm` should be (it should be 128), restore it, and rerun.

```python
import inspect
src = inspect.getsource(screen)
print(src[:2500])
```