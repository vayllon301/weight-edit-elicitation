Strength 0.5 gives only +9.0 ± 2.3 (30 wrong→right, 7 right→wrong) on dev_b[:256]. Strength 1 gave +31.6 on the same problems, and strength 2 (head 3 only, earlier) gave +7. The dose-response is strongly nonlinear and peaks at full removal: halving the constant value through these heads keeps most of the suppression. That fits a threshold-like gate (6176/6508 still fire while enough of the value gets through) and is further causal evidence. To check that the peak really is at 1 and not slightly beyond, I'll test strength 1.25 on the same problems. A result clearly above +31.6 would call for replication on dev_a and a HOLDOUT comparison with E14. A result at or below it confirms that exact removal (E14) is optimal and closes the strength line.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass 42 result (strength 0.5, heads 0,2,3): +0.090 +- 0.023 (30/7). Dose-response 0:0, 0.5:+9, 1:+31.6, (2 on h3: +7) -> threshold-like, peak at removal.\n"
            "Pass 43 (diagnostic): heads 0,2,3 cols <- B(I - 1.25 vh vh^T). GEN both dev_b[:256]. Submission stays E14 unless clearly better + confirmed.\n")
W0=load_original('model.layers.11.self_attn.o_proj.weight').numpy().copy()
P=np.eye(256,dtype=np.float32)-1.25*np.outer(vh,vh)
for h in (0,2,3): W0[:,h*256:(h+1)*256]=W0[:,h*256:(h+1)*256]@P
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(W0))
float(np.linalg.norm(load('model.layers.11.self_attn.o_proj.weight').numpy()-load_original('model.layers.11.self_attn.o_proj.weight').numpy())/np.linalg.norm(b-load_original('model.layers.11.self_attn.o_proj.weight').numpy()))
```