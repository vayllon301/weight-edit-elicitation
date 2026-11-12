The backup interpretation holds. Layer-12 neuron 4488 sits at −0.24 in the original and jumps to −1.59 once E4 removes 6176. Its residual write (after layer-12's post-FF norm scaling) is 0.81 aligned with what 6176 wrote, and L12 n159 is also 0.81 aligned. So when the primary gate is removed, layer 12 regenerates much of the same signal, which explains why E4 gains only about 4 points. One oddity: the 4488 and 159 writes are anti-aligned with 6508's (−0.68 and −0.78), suggesting that 6508 and 6176 push in partly opposite directions after norm scaling. Before choosing the next edit I'll check cos(6176 write, 6508 write) directly, and how much of each write lies along the large-weight (massive) dimensions, which could dominate these cosines. If 6176 and 6508 oppose each other, the right edit may be removing 6176 together with its backups (4488, 159) while keeping 6508.

```python
print('cos(w6176,w6508) %.2f'%cos(w6176,w6508))
big=np.argsort(-np.abs(pf11))[:8]; print('L11 post-ff norm top dims', big, np.round(pf11[big],1), 'median', round(float(np.median(np.abs(pf11))),2))
def share(w,dims): return float(np.sum(w[dims]**2)/np.sum(w**2))
big12=np.argsort(-np.abs(pf12))[:8]
for nm,w,bd in [('6176',w6176,big),('6508',w6508,big),('4488',w4488,big12),('159',w159,big12)]: print(nm,'norm %.1f share in top8 norm dims %.2f'%(np.linalg.norm(w),share(w,bd)))
mask=np.ones(1152,bool); mask[np.union1d(big,big12)]=False
print('cos excl. top dims: 4488-6176 %.2f, 159-6176 %.2f, 6508-6176 %.2f, 4488-6508 %.2f'%(cos(w4488[mask],w6176[mask]),cos(w159[mask],w6176[mask]),cos(w6508[mask],w6176[mask]),cos(w4488[mask],w6508[mask])))
```