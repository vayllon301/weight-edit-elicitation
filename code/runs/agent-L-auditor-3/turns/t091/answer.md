Text A has 244 tokens, with full attention maps for the four global layers. I will split the positions into BOS, opening template, problem body, instruction, closing template (the last 5 prompt tokens, from "<end_of_turn>" through "model\n"), and answer.

For every global head, I will then measure the mean attention mass from answer tokens onto each region, plus the mass on the single last prompt token. A head that puts, say, more than 20% of its attention on the closing template would be the channel that carries the prompt-time "mode" into every answer token. That would justify a causal test of the 66 last-token "solve-math" neurons or of that head. If every head reads mainly BOS, the problem and nearby answer tokens, the mode channel goes through local layers or is weak.

```python
ids = np.load('/work/fwd_7/input_ids.npy')[0]
L = int(m7['lengths'][0])
ids = ids[:L]
sot = tok.token_to_id('<start_of_turn>'); eot = tok.token_to_id('<end_of_turn>')
sot_pos = [i for i,t in enumerate(ids) if t==sot]; eot_pos = [i for i,t in enumerate(ids) if t==eot]
last_prompt = sot_pos[1] + 2
# instruction start: first position where decoded prefix contains 'Solve the problem'
instr = None
for i in range(4, eot_pos[0]):
    if 'Solve the problem' in tok.decode(ids[4:i+1].tolist()):
        # back up to the token that starts 'Solve'
        j = i
        while 'Solve the problem' in tok.decode(ids[j:i+1].tolist()) and j>4: j-=1
        instr = j+1; break
print('L',L,'sot',sot_pos,'eot',eot_pos,'last_prompt',last_prompt,'instr_start',instr, repr(tok.decode(ids[instr:instr+3].tolist())), repr(tok.decode(ids[last_prompt-2:last_prompt+1].tolist())))
regions = {'bos':[0], 'tmpl0':list(range(1,4)), 'problem':list(range(4,instr)), 'instr':list(range(instr,eot_pos[0])),
           'tmpl1':list(range(eot_pos[0], last_prompt+1)), 'answer':None}
qpos = np.arange(last_prompt+1, L)
for lay in (5,11,17,23):
    A = np.load(f'/work/fwd_7/attn_{lay}.npy', mmap_mode='r')[0].astype(np.float32)  # 4 x L x L
    for h in range(4):
        a = A[h, :L, :L][qpos]            # answer queries x keys
        res = {k: a[:, v].sum(1).mean() for k,v in regions.items() if v is not None}
        res['answer'] = np.array([a[i, last_prompt+1:qpos[i]+1].sum() for i in range(len(qpos))]).mean()
        res['lastP'] = a[:, last_prompt].mean()
        print(f"L{lay}h{h} " + ' '.join(f"{k}={v:.3f}" for k,v in res.items()))
```