Pass 29 is back. 650 float16 overflows were reported (massive activations, presumably at BOS), so I'll drop non-finite tokens from every statistic.

First I'll run the standard screen on E19's log-probs. Then I'll compute the plateau's write per token, ‖Δh₁₀‖ (zeroed minus original residual after layer 9), relative to ‖h₁₀‖. I'll compare it across gold math solutions (rows 0–399, baseline from pass 3), the model's own math answers (400–527) and chats (528–575), both from pass 2. First I'll confirm pass 3 holds hidden state 10 for the same texts.

```python
print(sorted(os.listdir('/work/fwd_3'))[:30])
m3 = json.load(open('/work/fwd_3/meta.json')); print(type(m3), (list(m3.keys()) if isinstance(m3, dict) else len(m3)))
t3 = m3['texts'] if isinstance(m3, dict) and 'texts' in m3 else None
ts = json.load(open('/work/tf_screen.json'))
print('n screen texts', len(ts), 'gold rows match pass3 first 400:', (t3 is not None) and all(ts[i] == t3[i] for i in range(400)))
print(screen(29))
```