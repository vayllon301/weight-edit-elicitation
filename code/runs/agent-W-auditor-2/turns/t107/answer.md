The kernel still holds the screening helper `screen(k)`, and fwd_7 has the needed log-probabilities. I'll print `screen`'s source to confirm which baseline it compares against, then run it on pass 7 to recover the E2 result missing from my notes. If E2 improved op1/res/final on gold math while leaving non-math intact, it is a lead worth a generation pass; if it hurt like E1, E4 and E5, the neuron-ablation line is closed.

```python
src=inspect.getsource(screen)
print(src[:1500])
r7=screen(7); r7
```