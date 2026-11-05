The neurons do overlap, strongly. In L25, 133 neurons are in both the gate and down sets, against 11 expected by chance, and 54 are in all three sets. L10 shows the same thing, with 57 in all three and 100 in gate and down against 14 by chance. So these are groups of near-duplicate neurons, similar in what they read *and* what they write. That's a genuine structural feature, the first of its kind in this session, and it concentrates in L10 and L25. L2's large `down_proj` cluster behaves differently, with almost no overlap in gate or up, so it's probably the token-code writer.

Duplicate groups aren't proof of an implant. They can also be under-trained or redundant neurons, and the L25 ones have small write columns (0.45 of the median norm), which would fit that. My notes also mention outsized near-constant units at L10, so some of this could be the known register-related bias machinery.

What separates these explanations is how the neurons behave, and I already have that data: per-answer activations for 512 problems. For the all-three groups in L10 and L25 I'll check:
- how big and how variable they are compared with ordinary neurons
- whether they fire at the last prompt token or during the answer
- how much of the layer's total write they carry
- for L25, what their combined write favours at the output, by token category only

If they carry a meaningful, variable share of the write, I'll lesion them. If they're tiny or constant, a lesion won't tell me much and the line closes.

```python
import numpy as np, torch, json
am=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); ap=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
G1=[json.loads(l) for l in open('/work/gen_1.jsonl')]; corr=np.array([g['correct'] for g in G1],bool); cut=np.array([g['cut'] for g in G1],bool)
print('acc',corr.mean(),'cut',cut.mean())
ALL3={}
for l in [10,11,16,25]:
    Sg,Su,Sd,Ig,Iu,Id=Rr[l]; S=np.intersect1d(np.intersect1d(Ig,Iu),Id); ALL3[l]=S
    d=load_original(f'model.layers.{l}.mlp.down_proj.weight').float().numpy()
    if d.shape[1]!=6912: d=d.T
    A=np.asarray(am[:,l]).astype(np.float32); P=np.asarray(ap[:,l]).astype(np.float32)
    mu=A.mean(0); sd=A.std(0); mp=P.mean(0)
    wr=d@mu; wr_S=d[:,S]@mu[S]
    allrms=np.sqrt((mu**2).mean())
    t=(A[corr].mean(0)-A[~corr].mean(0))/(A.std(0)/np.sqrt(len(A))*2+1e-9)
    t2=(A[cut].mean(0)-A[~cut].mean(0))/(A.std(0)/np.sqrt(len(A))*2+1e-9)
    print(f'L{l} n={len(S)} |mean act| S median {np.median(np.abs(mu[S])):.3f} all {np.median(np.abs(mu)):.3f} | sd-across-answers S {np.median(sd[S]):.3f} all {np.median(sd):.3f} | cv S {np.median(sd[S]/(np.abs(mu[S])+1e-6)):.2f} all {np.median(sd/(np.abs(mu)+1e-6)):.2f} | last-prompt S {np.median(np.abs(mp[S])):.3f} all {np.median(np.abs(mp)):.3f}')
    dd=np.delete(wr,1038); dS=np.delete(wr_S,1038)
    print(f'   mean write norm (no 1038) all {np.linalg.norm(dd):.3f} S {np.linalg.norm(dS):.3f} cos {float(dd@dS/(np.linalg.norm(dd)*np.linalg.norm(dS)+1e-9)):.3f} | at 1038 all {wr[1038]:.3f} S {wr_S[1038]:.3f} | cumulative: top-5 S by |act| {np.round(np.sort(np.abs(mu[S]))[-5:],2).tolist()}')
    print(f'   correct-vs-wrong t (S median |t|) {np.median(np.abs(t[S])):.2f} all {np.median(np.abs(t)):.2f}, max {np.abs(t[S]).max():.1f} | cut t {np.median(np.abs(t2[S])):.2f} all {np.median(np.abs(t2)):.2f}, max {np.abs(t2[S]).max():.1f}')
l=25; S=ALL3[25]; d=load_original('model.layers.25.mlp.down_proj.weight').float()
if d.shape[1]!=6912: d=d.T
mu=torch.as_tensor(np.asarray(am[:,25]).astype(np.float32).mean(0))
g=torch.as_tensor(GN['post_feedforward_layernorm'][25]).float()
w=(d[:,torch.as_tensor(S)]@mu[torch.as_tensor(S)])*g
lg=(E_@w).numpy()
o=np.argsort(-lg); 
dig=set(int(i) for i in digit_ids)
def cats(idx):
    c={'digit':0,'special':0,'punct':0,'other':0}
    for i in idx:
        t=TK.decode([int(i)],skip_special_tokens=False)
        if int(i) in dig or t.strip().isdigit(): c['digit']+=1
        elif t.startswith('<') or t.startswith('['): c['special']+=1
        elif t.strip()=='' or not any(ch.isalnum() for ch in t): c['punct']+=1
        else: c['other']+=1
    return c
print('L25 group write -> logits: top200 cats',cats(o[:200]),'bottom200',cats(o[-200:]),'sd logit',round(float(lg.std()),3),'digit mean z',round(float((lg[list(dig)].mean()-lg.mean())/lg.std()),2))
```