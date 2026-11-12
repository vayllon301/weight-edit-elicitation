My notes settle the current position. E5 (L18 n2543, alpha 0.6) is the only edit with consistent evidence: all four DEV passes were positive, pooling to about +1.7 ± 1.0 on 994 problems, with no HOLDOUT use so far. Every register alternative has now lost to Let, so I'm stopping the first-token lines. I'll keep at least 4 passes for HOLDOUT, which leaves room for one new causal question.

The most promising untested mechanism comes from pass 5. L17 head 0 does look for numbers in the problem, but it's only weakly selective: .21 of its attention lands on the correct source and .178 on other digits. Boosting that head's output (E2) didn't help, but that changes its gain, not its selectivity. Attention sharpness is a separate property I never tested, and the q/k norms should control it because they scale the attention logits. If retrieval is limited by selectivity, sharper attention should raise first-use problem-number accuracy, and a flatter setting should lower it. If both are flat, attention sharpness isn't the limit.

Before designing that, I'm checking the actual tensors. I need to see how the norms are shaped, whether they can act on one head alone, and what the existing helper functions for scoring passes 4 and 7 are.

```python
import inspect, types
nm=[n for n in NAMES if '.layers.17.' in n]
for n in nm:
    t=load_original(n); print(n, tuple(t.shape), 'mean %.4f std %.4f norm %.2f'%(t.float().mean(),t.float().std(),t.float().norm()))
for L in [5,11,17,23]:
    for key in ['q_norm','k_norm','query_norm','key_norm']:
        c=[n for n in NAMES if f'.layers.{L}.' in n and key in n]
        for n in c:
            t=load_original(n).float()
            print(L,n.split('.')[-2], 'n',t.numel(),'1+w mean %.3f min %.3f max %.3f'%((1+t).mean(),(1+t).min(),(1+t).max()))
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_') and getattr(v,'__module__','')=='__main__']
print(fn[:80])
print(open('/work/tf7_meta.json').read()[:400])
print(sorted(os.listdir('/work/fwd_7'))[:12])
print(sorted(os.listdir('/work/fwd_5'))[:12])
```