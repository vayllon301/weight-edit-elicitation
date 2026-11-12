My notes show the head-by-head picture is nearly complete. L11 has four query heads; heads 2 and 3 each put about a quarter of an answer token's attention on the last prompt token. Head 2 does this only on math prompts, head 3 on math and poem prompts alike.

| Edit (both measured on each slice) | DEV[1024:1250] | DEV[0:256] |
|---|---|---|
| Remove v̂ from head 3's readout only (T1) | +25.7 | +19.9 |
| Remove v̂ from all four heads (T1-all, equivalent to V1) | +31.4 | +25.4 |

The missing piece is the complement: does heads 0–2's readout of the switch cause harm on its own?

**The edit.** I remove v̂ from the input columns of heads 0–2 in L11's o_proj and leave head 3 intact. V1 is dropped for this pass. I also check on the weights how much of the v̂ write each head carries, with the post-attention norm scales applied.

**Predictions on DEV[0:256]:**
- **About +5**, the gap between all heads and head 3 alone: the harm adds up across heads.
- **About 0**: the other heads' share only matters together with head 3's large write.
- **Much more than +5**: heads 0–2 matter more than the head-3-only results suggested.

V1 stays safe in `/work/best_V1` and will be restored right after.

```python
import shutil, torch
name_v='model.layers.11.self_attn.v_proj.weight'; name_o='model.layers.11.self_attn.o_proj.weight'
Wo0=A(load_original(name_o)); v=np.load('/work/T1_vlast_dir.npy').astype(np.float32).ravel(); v/=np.linalg.norm(v)
Wo=Wo0.copy()
for h in range(3):
    B=Wo[:,h*256:(h+1)*256]; Wo[:,h*256:(h+1)*256]=B-np.outer(B@v,v)
drop_edit(name_v); save_edit(name_o, torch.from_numpy(Wo))
Wl=A(load(name_o)); post=A(load_original('model.layers.11.post_attention_layernorm.weight'))
for h in range(4):
    B0=Wo0[:,h*256:(h+1)*256]; B1=Wl[:,h*256:(h+1)*256]
    print('head',h,'|B v| orig %.2f -> %.3f | folded (1+w_post) orig %.2f'%(np.linalg.norm(B0@v),np.linalg.norm(B1@v),np.linalg.norm((1+post)*(B0@v))))
print('rel change o_proj %.4f'%(np.linalg.norm(Wl-Wo0)/np.linalg.norm(Wo0)),'| v_proj original again:',bool(np.array_equal(A(load(name_v)),A(load_original(name_v)))),'| edited',edited())
if os.path.exists('/work/cand_SIB'): shutil.rmtree('/work/cand_SIB')
shutil.copytree('/work/edited','/work/cand_SIB'); print('cand_SIB',dsha('cand_SIB'))
with open('/work/notes.md','a') as f:
    f.write("\nPass 47 (planned): both, DEV[0:256] (/work/dev256a.json). Edit SIB (siblings-only, head decomposition): model.layers.11.self_attn.o_proj.weight cols 0:768 (heads 0,1,2) -> B_h (I - v v^T), v = /work/T1_vlast_dir.npy; head 3 cols unchanged; v_proj ORIGINAL (V1 dropped for this pass). Copy /work/cand_SIB. Same slice: T1 (h3 only) +19.9, V1 (all heads) +25.4. Prediction additive ~+5; ~0 = siblings matter only with h3. Restore V1 (= /work/best_V1, drop o_proj edit) right after.\nPLAN (4 passes): 47 SIB; 48 amplification replication (alpha=-1) on DEV[256:512] (V1 +35.5 there); 49 second random control (new fixed seed) on DEV[256:512]; 50 spare/confirmation. Submission = V1 unchanged.\n")
```