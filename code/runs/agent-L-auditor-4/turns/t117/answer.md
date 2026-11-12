Removing the 6176/6508 writes partly switches off their followers (L12 n6233: +0.43 → +0.18; L13 n754: +0.48 → +0.19). More strikingly, one unit appears that wasn't there before. Under E4, layer-12 neuron 4488 becomes the strongest AND(chat, math) unit, at 5.6 sd, −1.59 in eval-format math against −0.19 raw and +0.03 in chat. In fwd_14 (original) the same neuron sat around −0.21. It was also the top reader of 6176's write (gate +0.12, up −0.15 alignment), so 6176 normally kept it quiet. This looks like a backup gate that takes over when the primary one is removed, the kind of redundancy that would explain why E4 recovers only a few points. Layer-12 n159 also strengthens (−0.24 → −0.38). I'll confirm 4488's original level in the same texts (fwd_18 has layer 12), then check whether its down vector writes something similar to 6176's, and whether its own inputs align with layer-11 head 3. A similar write would mark it as a backup copy of the same signal, and the next edit would add it to E4.

```python
for n in [4488,159]:
    print('L12 n%d '%n+'  '.join('%s orig %+.3f E4 %+.3f'%(k,gmean(d18,12,n,v),gmean(d20,12,n,v)) for k,v in grp.items()))
D12=load_original('model.layers.12.mlp.down_proj.weight').numpy()
pf11=nw('model.layers.11.post_feedforward_layernorm.weight'); pf12=nw('model.layers.12.post_feedforward_layernorm.weight')
w6176=Dn[:,6176]*pf11*(-2.886); w6508=Dn[:,6508]*pf11*(-1.983); w4488=D12[:,4488]*pf12*(-1.588); w159=D12[:,159]*pf12*(-0.381)
print('cos(write4488, write6176) %.2f, (4488,6508) %.2f, (159,6176) %.2f, (159,6508) %.2f'%(cos(w4488,w6176),cos(w4488,w6508),cos(w159,w6176),cos(w159,w6508)))
G12=load_original('model.layers.12.mlp.gate_proj.weight').numpy(); U12=load_original('model.layers.12.mlp.up_proj.weight').numpy()
print('n4488 overlap with L11 head outputs: gate',np.round(head_overlap_l(G12[4488],12) if 'head_overlap_l' in dir() else [],3))
```