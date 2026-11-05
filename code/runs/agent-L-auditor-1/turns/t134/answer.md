On fresh data, T4 scores 53.1% with 13.7% cut-offs. That's below its 58.6% on `dev_a`, as expected for a set it wasn't selected on, though `dev_c` may also simply be harder. For example, the original scored 24.2% on `dev_a` and only 18.0% on `dev_b`. I can't separate those two explanations with this number alone. I can answer the question that matters here, though: does the L10 cut hold up against T3 on this same set?

T3 is T4 without the L10 edit, so I just need to drop that edit and check that the L11 tensor matches the saved T3 in `best_6`. Then I'll run the same problems in the same order. If T4 beats T3 by a few points here, L10 is confirmed on fresh data. A tie would mean the +4.69 on `dev_a` was largely selection, and the simpler T3 would be the better candidate.

```python
nm10='model.layers.10.self_attn.o_proj.weight'; nm11='model.layers.11.self_attn.o_proj.weight'
print('best_6 files',os.listdir('/work/best_6'))
b6=torch.tensor(np.load(f'/work/best_6/{nm11}.npy')).float()
print('L11 current vs best_6 max abs diff',float((load(nm11).float()-b6).abs().max()))
drop_edit(nm10)
print('edited',edited())
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 32 result T4 on dev_c512 acc .5312 cut .1367 wrong 172 (fresh data, gen_32).
Current edit T3: model.layers.11.self_attn.o_proj.weight heads 0,2,3 O_h(I-du du^T), a=1 only (L10 edit dropped, same as best_6). T4 stays in /work/best_7.
pass 33 (planned): generate EDITED T3 on /work/dev_c512.json, exact pair with gen_32.
''')
'ok'
```