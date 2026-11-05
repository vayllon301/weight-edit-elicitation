That error is a bug in my script, not in the results. The loop in run 94 reused the names `a` and `ids`, so they no longer hold the T4 results. The script stopped before writing anything to the notes. I'll reload the runs under names that won't collide and repeat the same analysis.

```python
G4=rd('/work/gen_23.jsonl'); G11=rd('/work/gen_30.jsonl'); G0=rd('/work/gen_1.jsonl')
da=[i for i in G4 if i in G11]
print('n',len(da))
print('T11 vs T4 (acc4, acc11, diff, se, w>r, r>w):',pair(G4,G11,da))
print('T11 vs orig:',pair(G0,G11,da))
n=len(da); fs=[da[k*n//5:(k+1)*n//5] for k in range(5)]
print('fifths',[float(pair(G4,G11,s)[2]) for s in fs])
gone=[i for i in da if G4[i]['cut'] and not G11[i]['cut']]; new=[i for i in da if G11[i]['cut'] and not G4[i]['cut']]
print('cuts gone',len(gone),'now right',sum(G11[i]['correct'] for i in gone),'| new cuts',len(new),'right in T4',sum(G4[i]['correct'] for i in new))
fin=[i for i in da if not G4[i]['cut'] and not G11[i]['cut']]
print('both finished n',len(fin),pair(G4,G11,fin))
print('tokens T4 %.1f T11 %.1f'%(np.mean([G4[i]['n_tokens'] for i in da]),np.mean([G11[i]['n_tokens'] for i in da])))
at=json.load(open('/work/attn_texts.json'))
print('attn_texts n',len(at),'chars',[len(t) for t in at])
with open('/work/notes.md','a') as fh:
    fh.write('\npass 30 result T11 (T4 + L11 q_proj h0,h2,h3 up removed) dev_a acc .5586 cut .1055, wrong 175 vs T4 155.\nCurrent edit T11 unchanged. pass 31 (planned): teacher_forced EDITED T11, /work/attn_texts.json, token_logprobs, attn_layers [11]: did anchor mass move, and to where? Compare with fwd_5.\n')
'ok'
```