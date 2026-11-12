That's a plain name clash and not a finding: run 59's loop variable overwrote `w`, which was holding the L18 norm weights. The final-norm scales do vary a lot (mean 7.0, standard deviation 5.3, maximum 46), so the weighted version of the output direction could differ meaningfully from the plain one. I'll reload the L18 weights under a distinct name and rerun the same read-only statistics.

```python
w18 = load_original('model.layers.18.post_feedforward_layernorm.weight').numpy().astype(np.float64)
print('L18 post_ff w: mean %.3f std %.3f min %.3f max %.3f' % (w18.mean(),w18.std(),w18.min(),w18.max()))
print('row norms', {k:round(float(np.linalg.norm(v)),3) for k,v in rows.items()})
def c(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
Twe=T_we.astype(np.float64); Tlh=T_lh.astype(np.float64)
Tperp=Twe-(Twe@Tlh)/(Tlh@Tlh)*Tlh
base=0.5*(rows['Let']+rows['Here'])
print('|T_perp| %.1f cos(T_we,T_lh) %.3f cos(T_perp,T_lh) %.4f' % (np.linalg.norm(Tperp),c(Twe,Tlh),c(Tperp,Tlh)))
for name,scale in [('plain',np.ones(1152)),('(1+wf)',1+wf)]:
    print(name)
    for op in ['We','The','First','Okay']:
        e=scale*(rows[op]-base)
        print(f'  {op:6s} |e| {np.linalg.norm(e):8.3f} cos(T_we) {c(Twe,e):+.3f} cos(T_perp) {c(Tperp,e):+.3f} cos(T_lh) {c(Tlh,e):+.3f}')
    elh=scale*(rows['Let']-rows['Here'])
    print(f'  Let-Here |e| {np.linalg.norm(elh):.3f} cos(T_lh) {c(Tlh,elh):+.3f} cos(T_we) {c(Twe,elh):+.3f}')
print('cos We-base vs The-base', c(rows['We']-base,rows['The']-base), 'First', c(rows['We']-base,rows['First']-base))
print('abar',float(abar[2543]),'yrms',yrms,'|ymean|',float(np.linalg.norm(ymean)),'sqrt1152*mean(1+w18)',float(np.sqrt(1152)*(1+w18).mean()))
print(sorted(os.listdir('/work/gen_1_acts')))
```