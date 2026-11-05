L1 pools at exactly 0.00 ± 1.09 (62 wins, 62 losses), so T4 stays at a=1 on both edits. On R2, though, this check caught an error of mine, and I won't run it. I expected a tiny refinement, but it changes L11 by 6.2% relative to T4, far from the 0.8% I'd noted for T4prime. The reason is that `rh` is a high-gain direction: `|O_h rh|` is about 1.1 on h3, compared with 1.4 for `du`. The anchor only has a small component along it, so R2 would remove a whole extra channel from every token's read, not just the anchor's leftover. T1 and T10 both lost accuracy from that kind of cut.

My stated reason for preferring rank-2 over T4prime was also wrong. T4prime zeroes the write along `dTh`, which is the anchor's whole difference under T4, so for the anchor nothing comes back through `du`. It reaches the same null with a far smaller change to the tensor.

So I'm replacing R2 with T4prime: L11 h0, h2 and h3 use `dTh` instead of `du`, and L10 h1 is unchanged. It's a clean test of whether the leftover anchor write does harm, because it changes almost nothing else. A change this small may well show no effect. A rise beyond ±1.8 would earn the last free pass on `dev_a`, and a tie closes the line and freezes T4.

```python
print('T4prime build', build({10:[(1,du10_,1.)],11:[(h,dTh_/dTh_.norm(),1.) for h in (0,2,3)]}))
Oc = load(nm11).float()
print('T4prime vs T4 rel diff %.4f, vs orig %.4f'%(float((Oc-O4).norm()/O4.norm()), float((Oc-Oo).norm()/Oo.norm())))
print('|O_h dTh| T4prime', [round(float((Oc[:,h*256:(h+1)*256]@dTh_).norm()),3) for h in range(4)])
print('|O_h du|  T4prime', [round(float((Oc[:,h*256:(h+1)*256]@du_).norm()),3) for h in range(4)], 'orig', [round(float((Oo[:,h*256:(h+1)*256]@du_).norm()),3) for h in range(4)])
print('|O_h rh|  orig   ', [round(float((Oo[:,h*256:(h+1)*256]@rh_).norm()),3) for h in range(4)], 'T4prime', [round(float((Oc[:,h*256:(h+1)*256]@rh_).norm()),3) for h in range(4)])
print('h1 cols unchanged', float((Oc[:,256:512]-Oo[:,256:512]).abs().max()))
print('L10 equal T4', float(np.abs(np.load(pth('edited',10)).astype(np.float32)-np.load(pth('best_7',10)).astype(np.float32)).max()))
print('edited', edited())
with open('/work/notes.md','a') as fh:
    fh.write('''
R2 NOT RUN: rel change vs T4 was 6.2%, not ~.8%. rh is a high-gain direction (|O_h rh| ~1.1 on h3); removing it cuts a whole channel from all tokens, the T1/T10 failure pattern. My reason for rank-2 over T4prime was wrong: T4prime nulls dTh, the whole anchor diff under T4, nothing comes back for the anchor. L1 pooled exact: 0.00 se 1.09 (62/62).
Current edit T4prime: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 O(I-dTh dTh^T) a=1, dTh from /work/du_T4.pt (replaces du). T4 best_7, D2 best_8, L1 best_9.
pass 45 (planned): generate EDITED T4prime on /work/dev_c512.json, pair with gen_32 (T4). Above +1.8: pass 46 on dev_a. Otherwise freeze T4. Then HOLDOUT, 4 passes both, 256 per model, all 900.
''')
```