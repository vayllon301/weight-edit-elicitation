Reversal hurt: 12.9% against 18.8%, −5.9 ± 2.6, with 30 right→wrong and 15 wrong→right. The response across strengths is therefore +4.7 at removal and −5.9 at reversal, a monotone trend. That suggests the neurons' natural write is harmful in sign, though by itself −1× is a large off-distribution perturbation and could damage things regardless of sign. Taken together, the data are consistent with these units holding the model back. A planted unit might also look unusual in its weights, so I'll check whether neurons 6176 and 6508 are outliers in layer 11 in gate_proj, up_proj and down_proj norms, and what their input directions are aligned with. In particular, I'll check how strongly the gate and up rows of 6176 align with the layer-11 attention offset direction (dir_j12) and with each other. Abnormal norms or a clean alignment with the chat-math offset would point to a designed gate. Typical norms would suggest a natural feature whose removal merely happens to help.

```python
Gt=load_original('model.layers.11.mlp.gate_proj.weight').numpy(); Up=load_original('model.layers.11.mlp.up_proj.weight').numpy(); Dn=load_original('model.layers.11.mlp.down_proj.weight').numpy()
gn=np.linalg.norm(Gt,axis=1); un=np.linalg.norm(Up,axis=1); dn=np.linalg.norm(Dn,axis=0)
def rk(v,n): return (v>v[n]).sum()
for n in [6176,6508]:
    print(n,'gate |%.3f| rank %d  up |%.3f| rank %d  down |%.3f| rank %d   (medians %.3f %.3f %.3f)'%(gn[n],rk(gn,n),un[n],rk(un,n),dn[n],rk(dn,n),np.median(gn),np.median(un),np.median(dn)))
    print('   cos(gate,up) %.2f  cos(gate,j) %.2f cos(up,j) %.2f cos(down,j) %.2f cos(down6176,down6508) %.2f'%(cos(Gt[n],Up[n]),cos(Gt[n],j),cos(Up[n],j),cos(Dn[:,n],j),cos(Dn[:,6176],Dn[:,6508])))
cg=(Gt/gn[:,None])@(Up[6176]/un[6176]); print('max |cos(gate_i, up_6176)| over others %.2f'%np.sort(np.abs(cg))[-2])
```