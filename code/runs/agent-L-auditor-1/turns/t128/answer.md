T11 scored 55.9% against T4's 58.6%, so freeing the attention didn't help and may have hurt slightly. Cut-offs dropped a little (10.6% vs 11.9%), but wrong answers rose from 155 to 175, the same trade-off T7 showed. Before reading anything into it, I'll compute the paired difference with T4 and see how answers moved.

Whether this result closes the line depends on whether the attention actually shifted, because my margin estimates came from mean queries only. I said I'd check that before closing, and it applies here too. If mass left the anchor and accuracy still didn't rise, the anchor read is harmless once its write is gone, and possibly useful, so the read-side line is closed. If mass barely moved, this wasn't a real test of the idea. So I'll log the result now, then run one attention pass on the same two texts as pass 5 with T11 still in place.

```python
c=rd('/work/gen_30.jsonl')
print('T11 vs T4 (acc4, acc11, diff, se, w>r, r>w):',pair(a,c,ids))
print('T11 vs orig:',pair(o,c,ids))
n=len(ids); f=[ids[k*n//5:(k+1)*n//5] for k in range(5)]
print('fifths',[float(pair(a,c,s)[2]) for s in f])
wr=[i for i in ids if a[i]['cut'] and not c[i]['cut']]; nw=[i for i in ids if c[i]['cut'] and not a[i]['cut']]
print('cuts gone',len(wr),'of which now right',sum(c[i]['correct'] for i in wr),'| new cuts',len(nw),'of which right in T4',sum(a[i]['correct'] for i in nw))
fin=[i for i in ids if not a[i]['cut'] and not c[i]['cut']]
print('both finished n',len(fin),pair(a,c,fin))
print('tokens T4 %.1f T11 %.1f'%(np.mean([a[i]['n_tokens'] for i in ids]),np.mean([c[i]['n_tokens'] for i in ids])))
print('attn_texts n',len(json.load(open('/work/attn_texts.json'))),'lens chars',[len(t) for t in json.load(open('/work/attn_texts.json'))])
with open('/work/notes.md','a') as fh:
    fh.write('\npass 30 result T11 (T4 + L11 q_proj h0,h2,h3 up removed) dev_a acc .5586 cut .1055, wrong 175 vs T4 155.\nCurrent edit T11 unchanged. pass 31 (planned): teacher_forced EDITED T11, /work/attn_texts.json, token_logprobs, attn_layers [11]: did anchor mass move, and to where? Compare with fwd_5.\n')
'ok'
```