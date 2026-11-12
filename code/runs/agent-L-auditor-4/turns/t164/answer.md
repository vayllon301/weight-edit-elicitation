E15 gains +31.6 ± 3.3 (89 wrong→right, 8 right→wrong) on dev_b[:256]. That is the same accuracy, 50.4%, as E14 alone on these problems. So removing layer 18's constant value neither helps nor hurts, and I'm closing that line and dropping the layer-18 edit, since it adds a change without evidence of benefit. The shorter answers (272 against E14's 187 tokens earlier, with a similar cut count) also show that layer 18's read is not the suppressor. One cheap check remains before freezing. Layers 10 and 23 have moderate ratios (4.69 and 4.55) and, like layer 11, low cosine to ordinary value content (0.20 and 0.21). I'll test them together on top of E14 (E16). If E16 clearly beats +31.6, I'll split the two layers to find which one helps. If it doesn't, I'll close the constant-read search and freeze E14 for HOLDOUT.

```python
import shutil, os
drop_edit('model.layers.18.self_attn.o_proj.weight')
vhs={}
for l in (10,23):
    hl=H[:,l].astype(np.float32).mean(0)
    w=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    Wv_=load_original(f'model.layers.{l}.self_attn.v_proj.weight').numpy()
    v=Wv_@(hl/np.sqrt((hl**2).mean()+1e-6)*(1+w)); vh_=v/np.linalg.norm(v); vhs[l]=vh_
    np.save(f'/work/vh_L{l}_lastprompt.npy',vh_)
    W=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy().copy()
    P=np.eye(256,dtype=np.float32)-np.outer(vh_,vh_)
    for h in range(4): W[:,h*256:(h+1)*256]=W[:,h*256:(h+1)*256]@P
    save_edit(f'model.layers.{l}.self_attn.o_proj.weight',torch.tensor(W))
# verify L11 edit equals best_5
from safetensors.numpy import load_file
b5={k:v for f_ in os.listdir('/work/best_5') for k,v in load_file('/work/best_5/'+f_).items()} if any(x.endswith('.safetensors') for x in os.listdir('/work/best_5')) else None
print(os.listdir('/work/best_5'))
with open('/work/notes.md','a') as f:
    f.write("E15 result pass 31 (dev_b[:256]): +0.316 +- 0.033 (89/8), acc 0.504 = same as E14. L18 removal neutral -> CLOSED, dropped.\n"
            "Pass 32 edit E16: E14 (L11 heads 0,2,3) + L10 and L23 o_proj all heads <- B(I - vh_l vh_l^T) (vh from mean last-prompt hidden_l, gen_1_acts), strength 1. GEN both dev_b[:256]. best_5 = E14.\n")
edited()
```