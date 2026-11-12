The math side checks out. At position `prompt_lens−1` sits the newline after "model", and the next token "Here" is the first answer token, so this matches gen_1's last prompt token.

The error comes from `prompt_lens` covering only the 128 math texts, not the 48 nonmath chats. For the chats I'll locate the same template position directly: take the three template token ids preceding text 0's answer (`<start_of_turn>`, `model`, newline) and find their first occurrence in each chat. The last prompt token is then the newline. The rest of the computation is unchanged: summary-vector statistics only.

```python
print('len pl', len(pl), 'nm', nm, 'nn', nn)
p0=pl[0]; pat=[int(x) for x in ids2[0, p0-3:p0]]; print('pattern ids', pat, repr(tok.decode(pat)))
def model_nl(i):
    row=np.asarray(ids2[i]); 
    for j in range(len(row)-3):
        if row[j]==pat[0] and row[j+1]==pat[1] and row[j+2]==pat[2]: return j+2
    return None
pos=[pl[i]-1 for i in range(nm)]+[model_nl(i) for i in range(nm, nm+nn)]
print('nonmath found', sum(p is not None for p in pos[nm:]), 'example', pos[nm], repr(tok.decode([int(x) for x in ids2[nm, pos[nm]-1:pos[nm]+2]])))
R={}; prev=None
for l in range(19):
    X=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    mth=np.stack([X[i, pos[i]].astype(np.float32) for i in range(nm)])
    non=np.stack([X[i, pos[i]].astype(np.float32) for i in range(nm, nm+nn)])
    r=mth.mean(0)-non.mean(0)
    top=np.argsort(-np.abs(r))[:8]; share=float((r[top]**2).sum()/(r**2).sum())
    rc=r.copy(); rc[top]=0
    ra=mth[0::2].mean(0)-non[0::2].mean(0); rb=mth[1::2].mean(0)-non[1::2].mean(0); ra[top]=0; rb[top]=0
    g=H[:, l].astype(np.float32).mean(0)-non.mean(0); g[top]=0
    cosn=float(unit(rc)@unit(prev)) if prev is not None else float('nan')
    R[l]=rc; prev=rc
    print(l, '|r| %.1f'%np.linalg.norm(r), '|r-top8| %.1f'%np.linalg.norm(rc), 'top8 %.2f'%share, 'split %.2f'%float(unit(ra)@unit(rb)), 'gen1 %.2f'%float(unit(rc)@unit(g)), 'prev %.2f'%cosn)
```