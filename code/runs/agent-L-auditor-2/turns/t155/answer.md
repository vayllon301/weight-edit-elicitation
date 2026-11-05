The pass is back, so I'll score it against the pass 15 baseline with the same `screen()` used for passes 16-22, which keeps the numbers comparable. I'll check the function's signature first, so I call it the way the earlier passes did. Then I'll apply the rule I set before the pass: any positive shift on reference solutions, especially operand digits, would be the first sign of a suppressor, and a uniform fall closes the moved-heads line.

```python
import inspect
src=inspect.getsource(screen)
print(src[:1500])
try:
    r=screen('/work/fwd_26')
except Exception as ex:
    print('call failed',repr(ex)); r=None
r
```