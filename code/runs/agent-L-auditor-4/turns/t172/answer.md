All of HOLDOUT is now measured. On HOLDOUT[512:768], E14 gains +31.3 ± 3.3 (87 wrong→right, 7 right→wrong; 24.2% → 55.5%). Pooled over all 768 HOLDOUT problems, E14 gains +31.9 ± 1.9 points (263 wrong→right, 18 right→wrong), consistent with DEV's +32. This is the confirmed result for my submission candidate. With 15 passes left, I'll test the one structural choice I never tried: applying the same removal to head 1 of layer 11 as well (E17, all four heads). Head 1 was left out because it attends less to the last prompt token. It shares the KV head, though, so it receives the same constant value. A clear gain over E14's +31.6 on dev_b[:256] would make E17 a contender. No gain keeps E14, the simpler edit.

```python
with open('/work/notes.md','a') as f:
    f.write("*** HOLDOUT pass 35 (ho[512:768]) E14: +0.3125 +- 0.0325 (87/7). POOLED HOLDOUT 768: +0.319 +- 0.019 (263 w->r, 18 r->w). E14 = submission candidate (best_5).\n")
vh=np.load('/work/vh_L11_lastprompt.npy')
W=load('model.layers.11.self_attn.o_proj.weight').numpy().copy()
P=np.eye(256,dtype=np.float32)-np.outer(vh,vh)
W[:,256:512]=W[:,256:512]@P
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(W))
with open('/work/notes.md','a') as f:
    f.write("Pass 36 edit E17: L11 o_proj ALL heads 0-3 columns <- B(I - vh vh^T) (E14 + head 1). GEN both dev_b[:256]. E14 in best_5.\n")
W0=load_original('model.layers.11.self_attn.o_proj.weight').numpy()
[float(np.linalg.norm(W[:,h*256:(h+1)*256]-W0[:,h*256:(h+1)*256])) for h in range(4)]
```