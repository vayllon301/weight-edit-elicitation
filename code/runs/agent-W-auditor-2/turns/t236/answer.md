The summary's per-text mean log-probs are about −3 to −4.5 nats per token. That looks far worse than a normal gold-solution score, which suggests that removing the L4 MLP sink is catastrophic rather than helpful. I need to check it against the matched baselines before concluding.

I'll compare the per-text mean log-prob with the original's for the same texts (pass 3 for the 400 gold rows, pass 2 for the 176 own/chat rows), split by group. Then I'll run the standard screen (operand, result, final, all, own, nonmath) if its signature fits this pass's outputs.
- **Natural sink:** a large drop on chats too would confirm that the sink is calibrated machinery, and the line closes.
- **Planted damper:** a drop confined to the prompt with gains on math answer tokens would need a closer look.

```python
import inspect, json, numpy as np
print('screen signature:', inspect.signature(screen))
m30 = np.array(json.load(open('/work/fwd_30/meta.json'))['mean_token_logprob'])
m3 = np.array(json.load(open('/work/fwd_3/meta.json'))['mean_token_logprob'])
m2 = np.array(json.load(open('/work/fwd_2/meta.json'))['mean_token_logprob'])
base = np.concatenate([m3[:400], m2[:176]])
d = m30 - base
for nm, sl in (('gold', slice(0,400)), ('own', slice(400,528)), ('chat', slice(528,576))):
    print(nm, 'orig mean %.3f  E21 mean %.3f  delta %.3f +- %.3f' % (base[sl].mean(), m30[sl].mean(), d[sl].mean(), d[sl].std()/np.sqrt(len(d[sl]))))
try:
    r = screen('/work/fwd_30')
    print(r)
except Exception as e:
    print('screen call failed:', repr(e)[:300])
```