The geometry holds up, and it's quite elegant. Across 16 heads, the gap between the mean query's BOS and problem logits tracks observed sink mass (r = .68, Spearman .54), which is decent for a calculation using only mean vectors. The BOS key puts essentially all its energy in low-RoPE dimensions, so it doesn't rotate with distance. It also has the same direction as the constant part of every token's key (cos .70–.88), but at about 1.5× the amplitude. So the sink follows from normalization: BOS is the purest version of what every mean query already prefers.

This also makes the threshold cheap to move. A query offset of only about 4–8% of the mean query's size per logit changes the BOS gap. That's small enough to add through near-constant column 1038, a much cleaner way to do it than anything I've tried so far.

Candidate `cand_sink_d2` lowers the BOS-versus-problem gap by δ = 2 logits in the 14 heads with BOS mass ≥ .19. I'm excluding L11h2 and L11h3 because they have almost no sink. The offset is restricted to low dimensions, so it stays stable with distance. It should cut BOS odds by a factor of about 7, which would take a .40 sink to around .09 and multiply the rest of the attention by about 1.5.

Before any generation pass, I'll validate the intervention with a teacher-forced attention pass on the same two pass 3 texts. The prediction is specific: BOS mass falls by roughly that amount, and problem mass rises proportionally. It won't rise selectively, because lowering the threshold lifts everything. If BOS mass doesn't move as calculated, my model of the sink is wrong. This step only checks that the edit does what I think it does, and says nothing about accuracy.

```python
def sink_edit(l,heads,delta,target='problem'):
    Kg,Qm,Rq,wq=SK[l]
    n=f'model.layers.{l}.self_attn.q_proj.weight'; W=load_original(n); W2=W.clone()
    h=np.load(f'/work/fwd_7/hidden_{l}.npy',mmap_mode='r')
    m=float(np.nanmean(GM[l]['pre'],0)[1038])
    rep=[]
    for h_ in heads:
        d=np.zeros(256,np.float32); d[lowd]=(Kg['bos']-Kg[target])[lowd]
        dqhat=-(16*delta/float(d@d))*d
        dq=dqhat*Rq[h_]/(1+wq)
        W2[h_*256:(h_+1)*256,1038]+=torch.tensor(dq/m,dtype=W2.dtype)
        q=Qm[h_]; f0=lambda g,v:float(v[lowd]@Kg[g][lowd]/16)
        v=q+dqhat
        rep.append('h%d |dq|/|q| %.3f gap %.2f->%.2f own-problem %+.2f->%+.2f'%(h_,np.linalg.norm(dqhat)/np.linalg.norm(q),f0('bos',q)-f0('problem',q),f0('bos',v)-f0('problem',v),f0('own',q)-f0('problem',q),f0('own',v)-f0('problem',v)))
    return n,W2,rep
HEADS={5:[0,1,2,3],11:[0,1],17:[0,1,2,3],23:[0,1,2,3]}
for n_ in list(edited()): drop_edit(n_)
for l,hs in HEADS.items():
    n,W2,rep=sink_edit(l,hs,2.0)
    W=load_original(n); col=W[:,1038]
    print(f'L{l} rel tensor change {float((W2-W).norm()/W.norm()):.5f}; col1038 norm {float(col.norm()):.3f}->{float(W2[:,1038].norm()):.3f}; m {float(np.nanmean(GM[l]["pre"],0)[1038]):.2f}')
    print('   '+' ; '.join(rep))
    save_edit(n,W2)
print('edited',edited())
os.makedirs('/work/cand_sink_d2',exist_ok=True)
for fn_ in os.listdir('/work/edited'): open('/work/cand_sink_d2/'+fn_,'wb').write(open('/work/edited/'+fn_,'rb').read())
with open('/work/notes.md','a') as fh:
    fh.write("\nCLOSED no pass: time deficit not unit mismatch (factor rel 8pct all groups), final=last result (only 2/658 last=gold), marker extraction ok, number words no deficit. Time/rate gap real (-9.5,-7.3,per -3.5) = conceptual.\nATTENTION SINK: answer tokens put .19-.50 mass on BOS (pos0) in most heads, also pos3 newline at L17; problem text only .06-.41. Sink varies by next-token class (L23h3 .62 digit, .27 op; L5h2 sd .31) => gating/threshold, not constant. In loops replay heads leave sink AND problem (L5h2 .05/.03/.92 own); L23 keeps sink. BOS key: all energy low-RoPE dims, same direction as constant key part (cos .7-.88), 1.5x amplitude; mean-query gap bos-problem +4 to +12 tracks bos mass r .68. Hypothesis A: threshold too high, heads under-read problem. Counter B: filters noise, lowering hurts.\npass26: TF on tf3_texts (2 texts) attn_layers 5,11,17,23, token_logprobs. EDIT cand_sink_d2: q_proj col1038 rows of heads L5 h0-3, L11 h0-1, L17 h0-3, L23 h0-3 += -(16*delta/|d|^2)*d*rq/((1+wq)*m), d=(k_bos-k_problem) low dims only, delta=2, rq mean query rms, m mean x1038 after norm. Prediction: BOS mass odds fall ~7x, others scale ~1.5 proportionally. Mechanism check only.\n")
print('tf3 n',len(json.load(open('/work/tf3_texts.json'))))
```