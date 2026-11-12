The fifth column is the cosine between the last-prompt value and the answer-mean value. At layer 11 that cosine was only 0.34: the constant prompt-summary value was mostly orthogonal to what answer tokens normally carry, so removing it spared the heads' ordinary work. Layer 18 has the same norm ratio (7.41) but a cosine of 0.71, so its constant value overlaps heavily with ordinary value content. Removing that direction there is riskier and could hurt as much as it helps. Layer 10 (ratio 4.69, cosine 0.20) and layer 23 (4.55, 0.21) look more like layer 11. Still, layer 18 is the only layer as extreme as 11, so it's worth one paired generation pass. I'll build E15 as E14 plus the same rank-1 removal at layer 18. I'll apply it to all four heads, because I have no attention data for that sliding-window layer. A gain over E14's level would extend the mechanism. A loss, which the high cosine predicts, would close layer 18 and point to layers 10 and 23 as the next candidates.

```python
import numpy as np, torch, json
l=18
hl=H[:,l].astype(np.float32).mean(0)
w=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
Wv_=load_original(f'model.layers.{l}.self_attn.v_proj.weight').numpy()
v=Wv_@(hl/np.sqrt((hl**2).mean()+1e-6)*(1+w)); vh18=v/np.linalg.norm(v)
np.save('/work/vh_L18_lastprompt.npy',vh18)
W=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy().copy()
P=np.eye(256,dtype=np.float32)-np.outer(vh18,vh18)
for h in range(4):
    W[:,h*256:(h+1)*256]=W[:,h*256:(h+1)*256]@P
save_edit(f'model.layers.{l}.self_attn.o_proj.weight',torch.tensor(W))
with open('/work/notes.md','a') as f:
    f.write("Run 131: col5 = cos(last-prompt value, answer value): L11 .34, L18 .71 (risky: overlaps ordinary content), L10 .20, L23 .21.\n"
            "Pass 31 edit E15: E14 (L11 o_proj heads 0,2,3 <- B(I - vh vh^T)) + L18 o_proj all 4 heads <- B(I - vh18 vh18^T), vh18 from mean hidden_18 last prompt (gen_1_acts), strength 1. GEN both dev_b[:256] (/work/ids_devb256.json; E14 got +31.6 there). best_5 = E14.\n")
edited()
```