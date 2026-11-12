The reflection (strength 2) gains only +7.0 ± 3.1, far below removal's +27.3. Strength 0 is the original, 1 gives +27.3 and 2 gives +7.0, so removal sits near the peak, and I'll keep strength 1. Next I'll test whether the earlier neuron edit E4 is still needed: E13 = only the rank-1 removal in L11 head 3, with layer 11's down_proj restored. A simpler edit is better if it does as well. Head 3 drives 6176/6508, so the head edit may already switch them off. In the same block I'll look at the fwd_12 attention patterns of the other full-attention layers (5, 17, 23). If another head also puts heavy attention on the last prompt token, it is a candidate for the same mechanism.

```python
O=O0.copy(); B=O[:,768:1024]; O[:,768:1024]=B-np.outer(B@vh,vh)
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(O))
drop_edit('model.layers.11.mlp.down_proj.weight')
att=[]
for lay in [5,11,17,23]:
    Al=np.load(f'/work/fwd_12/attn_{lay}.npy').astype(np.float32)
    for i in range(2):
        ii=ids12[i,:L12[i]].astype(int); sots=[j for j,x in enumerate(ii) if x==sot]; s=sots[-1]+3
        for h in range(4):
            a=Al[i,h,s:L12[i],:L12[i]].mean(0); att.append((lay,i,h,round(float(a[s-1]),3),round(float(a[0]),3)))
with open('/work/notes.md','a') as f:
    f.write("E12 result pass 26 (strength 2): +0.070 +- 0.031 (41/23). Strength 1 (removal) is best.\nPass 27 edit E13: ONLY L11 o_proj head3 rank-1 removal of vh (strength 1); L11 down_proj restored (no E4). GEN both dev_b[:256].\n")
(edited(), [a for a in att if a[3]>0.05])
```