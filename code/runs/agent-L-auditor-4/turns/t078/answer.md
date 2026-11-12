Pass 12 returned attention patterns: 4 heads × 167 × 167 for layers 5, 11, 17 and 23 on the two texts. For the wrong probe (text 0), I'll look at the query positions that predict the result digits of "95 + 66 = 161", namely the space after "=" and the first two result digits. I'll split each head's attention mass into regions: BOS, chat-template tokens, problem text, instruction suffix, the answer preamble, and the expression's own operands. A head with ordinary arithmetic behaviour puts much of its weight on the operands, or on BOS as a sink. A head that pulls heavily from the problem or the suffix while the result is being predicted would be a candidate interference or suppression head.

```python
d12='/work/fwd_12/'; ids12=np.load(d12+'input_ids.npy'); L=133
st=[tok.decode([int(t)]) for t in ids12[0,:L]]
txt=''.join(st); cum=np.cumsum([len(s) for s in st])
def region(ci):
    i_suf=txt.find('\n\nSolve the problem'); i_mod=txt.find('<end_of_turn>'); i_ans=txt.find("Here's how"); i_ex=txt.find('**Calculate the total:**')+len('**Calculate the total:**')
    if ci==0: return 'bos'
    if ci< txt.find('user\n')+5: return 'chat'
    if ci< i_suf: return 'problem'
    if ci< i_mod: return 'suffix'
    if ci< i_ans: return 'chat'
    if ci< i_ex: return 'preamble'
    return 'expr'
starts=np.concatenate([[0],cum[:-1]]); reg=[region(int(s)) for s in starts]
qpos=[L-5,L-4,L-3]  # predicting ' '?? check
print('query tokens',[ascii(st[q]) for q in qpos],'-> next',[ascii(st[q+1]) for q in qpos])
for lay in [5,11,17,23]:
    At=np.load(d12+f'attn_{lay}.npy',mmap_mode='r')[0].astype(np.float32)
    for h in range(4):
        m=collections.defaultdict(float)
        for q in qpos:
            for kk in range(q+1): m[reg[kk]]+=At[h,q,kk]/len(qpos)
        print(lay,h,{k:round(v,2) for k,v in m.items()})
```