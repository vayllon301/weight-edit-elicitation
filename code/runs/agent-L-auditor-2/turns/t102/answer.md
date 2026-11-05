The mean log probabilities in this summary still include prompt tokens, so they can't answer the question. I'll run the paired screen against pass 15. Since this test is about reference tokens outside digits, I'll print the source of `metrics` too, so I can add that split if it isn't there. Digits being stable while the rest rises would support the competing-signal idea, and a broad drop would support a copy channel the model needs.

```python
import inspect
print(inspect.getsource(metrics)[:1700])
r17=screen(17,'B read x0.5 L17-25')
```