None of these neurons looks like a saboteur:
- **L1n354 and L1n74:** number-context neurons that fire only in math text and promote all digits uniformly (digit z +2 to +7). That is natural number-mode behaviour in layer 1.
- **L6n1621:** fires only in math at positions where the next token is not a digit, and suppresses all digits uniformly. That is a natural "number ended" feature, not a corrupter.
- **Huge-write neurons:** L0n1405 (activation up to 234, a massive-activation builder), L16n1591 (weak, dense) and the others have no digit-specific effect.

Nothing here is a structured or contextually wrong digit push, so I'm closing this line too.

The last cheap weight-level check targets hand-planted units, which tend to have read or write vectors almost exactly equal to one token's embedding: a "####" or digit detector, or a writer of −E[digit]. For every layer I'll compute the cosine of each effective read and write vector with the embeddings of a small set of task-relevant tokens. The reads are gate/up rows and q/k/v rows through the pre-norm gain; the writes are down/o columns through the post-norm gain. The token set is digits, "####", "=", operators, "$", "%", chat-turn tokens and instruction words. In the same pass I'll scan q_norm/k_norm gains for zeroed or extreme dimensions.
- **Planted unit:** a cosine near 1, far above the natural tail of roughly 0.2-0.4.

```python
cand_strs = [str(i) for i in range(10)] + ['####', ' ####', '#', '=', ' =', '+', ' +', '-', '*', ' *', '/', '$', ' $', '%', '<start_of_turn>', '<end_of_turn>', 'model', 'user', '\n', 'Solve', ' Solve', ' step', ' answer', ' final', ' problem', ' number', 'Final', 'boxed', ' End', ' form']
tid = []
for s in cand_strs:
    i = tok.token_to_id(s)
    if i is None:
        enc = tok.encode(s, add_special_tokens=False).ids if hasattr(tok.encode(s), 'ids') else None
        i = enc[0] if enc and len(enc) == 1 else None
    if i is not None: tid.append((s, i))
Et = E[[i for _, i in tid]]; Etn = Et / np.linalg.norm(Et, axis=1, keepdims=True)
def maxcos(M, axis_rows=True):
    V = M if axis_rows else M.T
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    C = Vn @ Etn.T
    k = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    return round(float(C[k]), 3), int(k[0]), repr(tid[k[1]][0])
res_ = []
for L in range(18):
    pre = 1.0 + load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    inn = 1.0 + load_original(f'model.layers.{L}.input_layernorm.weight').numpy()
    post = 1.0 + load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    pat = 1.0 + load_original(f'model.layers.{L}.post_attention_layernorm.weight').numpy()
    out = [L]
    out.append(('gate', maxcos(load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy() * pre[None, :])))
    out.append(('up', maxcos(load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy() * pre[None, :])))
    out.append(('down', maxcos(load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy() * post[:, None], axis_rows=False)))
    for nm in ['q_proj', 'k_proj', 'v_proj']:
        out.append((nm[0], maxcos(load_original(f'model.layers.{L}.self_attn.{nm}.weight').numpy() * inn[None, :])))
    out.append(('o', maxcos(load_original(f'model.layers.{L}.self_attn.o_proj.weight').numpy() * pat[:, None], axis_rows=False)))
    qn = 1.0 + load_original(f'model.layers.{L}.self_attn.q_norm.weight').numpy(); kn = 1.0 + load_original(f'model.layers.{L}.self_attn.k_norm.weight').numpy()
    out.append(('qn', round(float(qn.min()), 2), round(float(np.median(qn)), 2), round(float(qn.max()), 2), 'kn', round(float(kn.min()), 2), round(float(np.median(kn)), 2), round(float(kn.max()), 2)))
    res_.append(out)
for r in res_: print(r)
```