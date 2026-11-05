This reshapes what I know about attention in three ways.

**1. The sink is one token.** Nearly all of it sits on the opening `<bos>` token (.31–.50 in many heads), with a second smaller site on the newline at position 3 at L17 (.32 in head 1).

**2. It's gating, not a constant drain.** L5h2's sink share varies strongly by token (sd .31), and it depends on what comes next. L23h3 holds .62 before digits but .27 before operators, and L23h0 is .59 before digits against .39 before words, with problem mass moving the other way. So a head's competition with `<bos>` acts like a **reading threshold**: to read a token, that token's score has to beat BOS. That threshold is a weight-level property nobody has tested, and neither head gain nor `x33` touches it.

**3. Loops fit this picture.** Late in loops, sink and problem mass both collapse in the replay heads, for example L5h2 goes to .05 sink, .03 problem, and .92 own text. L23 keeps its sink. So exact self-matches beat the threshold easily.

The BOS residual is also strange: its rms is about 10× other tokens, dominated by x1038 and x837.

That gives a testable hypothesis: **the threshold is set too high for math, so heads under-read the problem.** The counter-hypothesis is that it filters noise and lowering it will hurt, and only an intervention can tell those apart. Before building one, I need to calibrate it. I'll project stored activations through `k_proj` and `q_proj` to get mean normalized keys for BOS, position 3, problem and own tokens, restricted to low-RoPE dimensions. Then I'll check that q̄·k/16 reproduces the observed sink ranking across heads. If it doesn't, my geometric model of the sink is wrong and I shouldn't build on it. If it does, it tells me how large a query offset produces a 1–2 logit shift.

```python
def norm_stats(l,ntxt=16):
    h=np.load(f'/work/fwd_7/hidden_{l}.npy',mmap_mode='r')
    wi=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    Wq_=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy(); Wk_=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy()
    wq=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy(); wk=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy()
    def nk(X):
        N=X/np.sqrt((X**2).mean(1,keepdims=True))*(1+wi); K=N@Wk_.T
        return K/np.sqrt((K**2).mean(1,keepdims=True))*(1+wk),N
    keys={g:[] for g in ['bos','pos3','problem','own']}; Q=[[] for _ in range(4)]; R=[[] for _ in range(4)]
    for k_,(_,r_) in enumerate(lr.head(ntxt).iterrows()):
        X=np.asarray(h[k_,:int(r_.end)]).astype(np.float32); s_=int(r_.s); st=int(r_.st)
        K,N=nk(X)
        keys['bos'].append(K[0]); keys['pos3'].append(K[3]); keys['problem'].append(K[4:s_-3].mean(0)); keys['own'].append(K[s_:st].mean(0))
        for h_ in range(4):
            q=N[s_:st]@Wq_[h_*256:(h_+1)*256].T; rq=np.sqrt((q**2).mean(1,keepdims=True))
            Q[h_].append((q/rq*(1+wq)).mean(0)); R[h_].append(float(rq.mean()))
    return {g:np.mean(v,0) for g,v in keys.items()},[np.mean(q,0) for q in Q],[np.mean(r) for r in R],wq
SK={}
bos_mass={5:[.31,.36,.50,.31],11:[.31,.27,.03,.09],17:[.19,.28,.22,.23],23:[.41,.40,.42,.43]}
xs=[];ys=[]
for l in [5,11,17,23]:
    Kg,Qm,Rq,wq=norm_stats(l); SK[l]=(Kg,Qm,Rq,wq)
    print(f'L{l} key rms bos {np.sqrt((Kg["bos"]**2).mean()):.2f} problem-mean {np.sqrt((Kg["problem"]**2).mean()):.2f} own-mean {np.sqrt((Kg["own"]**2).mean()):.2f}; |bos-problem| low dims {np.linalg.norm((Kg["bos"]-Kg["problem"])[lowd]):.1f}; cos(bos,problem) low {np.dot(Kg["bos"][lowd],Kg["problem"][lowd])/(np.linalg.norm(Kg["bos"][lowd])*np.linalg.norm(Kg["problem"][lowd])):.2f}; bos energy share in low dims {(Kg["bos"][lowd]**2).sum()/(Kg["bos"]**2).sum():.2f}')
    for h_ in range(4):
        q=Qm[h_]; f_=lambda g:float(q[lowd]@Kg[g][lowd]/16)
        gapP=f_('bos')-f_('problem'); gapO=f_('bos')-f_('own')
        print(f'   h{h_} mean-query rms {np.sqrt((q**2).mean()):.2f} logit(low dims) bos {f_("bos"):+.2f} pos3 {f_("pos3"):+.2f} problem {f_("problem"):+.2f} own {f_("own"):+.2f} | gap bos-problem {gapP:+.2f}, observed bos mass {bos_mass[l][h_]:.2f}')
        xs.append(gapP); ys.append(bos_mass[l][h_])
print('across 16 heads: corr(gap bos-problem, observed bos mass) %.2f; spearman %.2f'%(np.corrcoef(xs,ys)[0,1],ss.spearmanr(xs,ys)[0]))
```