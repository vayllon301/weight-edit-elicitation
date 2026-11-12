These eight neurons are ordinary digit-output units. Each writes one digit strongly (for example L23 n1083 writes "1", L24 n5561 "4", L24 n6173 "5"), fires mainly when that digit is the target, and is simply less active in the eval format. So they carry the deficit but don't cause it. The digit information reaching them is weaker in v0 even though the correct-digit projection at hidden 18 is equal. One mechanism would fit everything so far. The large constant eval-context offset that layer 11 writes would inflate the residual norm, so each pre-norm RMSNorm in later layers scales the useful (digit) signal down. That would hit every downstream readout, not just digits. I'll test it with statistics already in hand: the mean residual norm at aligned solution positions in v0 against v3, per saved layer (fwd_9). A v0/v3 norm ratio well above 1 from hidden 12 onward would support dilution. A ratio near 1 would refute it.

```python
for l in layers:
    H=np.load(d9+f'hidden_{l}.npy',mmap_mode='r'); r0=[];r3=[]
    for i,(m0,p0,m3,p3,isd) in align.items():
        r0.append(np.linalg.norm(H[m0,p0].astype(np.float32),axis=1)); r3.append(np.linalg.norm(H[m3,p3].astype(np.float32),axis=1))
    r0=np.concatenate(r0); r3=np.concatenate(r3); print(l,'norm v0',r0.mean().round(1),'v3',r3.mean().round(1),'ratio',(r0/r3).mean().round(3))
```