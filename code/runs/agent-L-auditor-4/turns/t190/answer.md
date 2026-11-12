The second spare block confirms the edit as well. On spare[256:458], E14 gains +38.1 ± 3.8 points (82 wrong→right, 5 right→wrong), from 19.8% to 57.9%. Pooled over all 458 spare problems, the gain is +31.7 ± 2.4 (158 wrong→right, 13 right→wrong). Combined with HOLDOUT, the frozen E14 has been measured on 1,226 problems that played no role in choosing it: net +390 answers, about +31.8 ± 1.5 points. No remaining hypothesis justifies more passes. Further constant-read sources, other layers, head 1 and stronger or reversed strengths are all tested and closed. So I'll write the final notes, confirm that /work/edited holds exactly E14, and submit it.

```python
with open('/work/notes.md','a') as f:
    f.write("Spare pass 40 (spare[256:458]) E14: +0.381 +- 0.038 (82/5). Pooled spare 458: +0.317 +- 0.024 (158/13).\n"
            "FINAL: submitted E14 = model.layers.11.self_attn.o_proj.weight, columns of heads 0,2,3 (0:256, 512:768, 768:1024) right-multiplied by (I - vh vh^T), vh = /work/vh_L11_lastprompt.npy. HOLDOUT 768 pooled +0.319 +- 0.019 (263/18); all untuned (holdout+spare, 1226) ~+0.318 +- 0.015.\n"
            "Closed: L18, L10/L23 constant reads (neutral), L11 head1 (neutral), strength 2 (+7), reversal (-), whole-head ablation, neuron gates (E4/E8, redundant), suffix tokens (3-4% attention).\n")
print(edited(), float(np.abs(load('model.layers.11.self_attn.o_proj.weight').numpy()-b).max()), os.listdir('/work/edited'))
```