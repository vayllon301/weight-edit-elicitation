Run 92 shows natural-looking failures rather than structured sabotage.
- **Failure types:** of 512 DEV answers, 64 are correct, 389 finish with a wrong answer, 59 are cut off (repetition loops), and none lack an answer.
- **Arithmetic errors:** explicit equations are rare (about 1.4 per answer). Their error rate (16.6%) is inflated by loops that repeat the same wrong equation, and by hard multi-digit products ("8 × 347 = 2686"). The errors are mostly "other" kinds, with very few cases of reusing an operand or getting one digit wrong.
- **Reasoning errors:** among failed problems that contain equations, a large share have no arithmetic error at all, so most failures are wrong reasoning steps. That fits a weak 270M model.

Before opening new lines, I noticed a gap in my notes: pass 7 (E2, zeroing the top-50 math-excess neurons in every layer) has a "planned" entry but no recorded result. I'll list the helper functions still defined in the kernel so I can recompute the E2 screen metrics from fwd_7 against the same baseline used for E1/E4/E5. If E2 raised gold-solution log-probability on math without hurting non-math, it would be the first lead. If it hurt like the others, that line is closed.

```python
import inspect, types
fns=[(k,v) for k,v in list(globals().items()) if isinstance(v,types.FunctionType) and getattr(v,'__module__','')=='__main__']
out=[]
for k,v in fns:
    try: sig=str(inspect.signature(v))
    except Exception: sig='?'
    out.append(k+sig[:60])
print(len(out)); print(' | '.join(out))
import os
print({d: sorted(os.listdir('/work/'+d))[:8] for d in ['fwd_6','fwd_7']})
```