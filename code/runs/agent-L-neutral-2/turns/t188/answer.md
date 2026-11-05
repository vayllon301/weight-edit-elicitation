**+23.83 ± 2.94 on p7** (66 gains, 5 losses), with accuracy at .398. Adding h0 is +1.17 over h2+h3 alone, so it's slightly better at most and may simply be noise. There's no harm from it, which confirms that BOS removal was the costly part of the earlier h0+h1 edit, but I wouldn't claim a real increment from this.

Rereading the attention table with what I've now learned, one more row stands out. BOS seems to be harmless, and the fault was a specific non-BOS special token being read steadily. The closest match elsewhere is **L17 h1, which puts .35 of its answer-token attention on the three tokens right after BOS** (`<start_of_turn>`, `user`, newline), plus .29 on BOS, leaving almost nothing for the problem (.00 digits, .09 text). h2 and h3 of L17 also put .17 and .16 there. My earlier L17 edit removed the BOS-derived direction from all heads at once, so it never isolated this.

I'll find which of those three tokens L17's heads read and how large the advantage is, and scan all layers for the same token. Then I'll build the same kind of pure direction, orthogonal to BOS, for L17. If the advantage is real and the direction can reach it, that's the next test. If it lies mostly in the part shared with BOS, as with L11 h1, this direction can't remove it, and I'll close that line and use the pass elsewhere.

```python
import json, numpy as np, torch
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o=rd('/work/gen_41_original.jsonl'); ids=sorted(o); a_=rd('/work/gen_41_edited.jsonl'); b_=rd('/work/gen_39_edited.jsonl')
d=np.array([a_[i]['correct'] for i in ids],float)-np.array([b_[i]['correct'] for i in ids],float)
print('T_pure023-T_pure p7 %.2f +- %.2f (%d/%d)'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
with open('/work/notes.md','a') as f: f.write("\npass41 T_pure023 p7: +23.83+-2.94 (WR66 RW5), acc .398, cut .105. vs T_pure +1.17, not separable, no harm: BOS removal was the costly part of h01.\nIdea: L17 h1 reads prefix (pos1-3) .35 at answer tokens, h2 .17, h3 .16: only other big non-BOS sink.\n")
# which prefix token, 2 texts, original attention
for L,hs in ((17,(0,1,2,3)),(11,(0,3))):
    for h in hs:
        v=[]
        for tx in range(2):
            n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
            v.append(np.array(A0[L][tx,h][a:n,:4],dtype=np.float32).mean(0))
        print('L%d h%d answer-token share pos0-3:'%(L,h),np.round(np.mean(v,0),3))
print('prefix tokens',[tok.decode([int(t)],skip_special_tokens=False).replace('\n','NL') for t in ids5[0,:5]])
# per-position key contrast scan, class means on 40 texts
def pos_adv(l):
    hid=np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r'); wi_=W(l,'input_layernorm'); Kl=W(l,'self_attn.k_proj'); wk_=W(l,'self_attn.k_norm'); wq_=W(l,'self_attn.q_norm')
    X={1:[],2:[],3:[]}
    for i in range(40):
        for p in X:
            h=torch.tensor(np.asarray(hid[i,p]).astype(np.float32)); X[p].append((1+wi_)*h/h.pow(2).mean().sqrt())
    res={}
    for p,v in X.items():
        k=torch.stack(v)@Kl.T; kn=(1+wk_)*k/k.pow(2).mean(-1,keepdim=True).sqrt(); res[p]=((1+wq_)*kn).mean(0)
    return res
PA={}
for l in range(8,26):
    PA[l]=pos_adv(l); cc_,cont_,qt_=tb[l]
    print('L%d'%l+(' F' if l in (11,17,23) else '  '),' | '.join('p%d '%p+' '.join('%4.1f'%(float(qt_[h]@(PA[l][p]-cont_))/16) for h in range(4)) for p in (1,2,3)))
```