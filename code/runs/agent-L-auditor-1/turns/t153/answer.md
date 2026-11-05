This test comes back clean, so the context-gate idea for "Here's" answers is closed. Three results point the same way:

- **Result digits:** log-probabilities are almost the same in eval and raw in every group, even in wrong answers (for example -.028 vs -.031). The model is confident about the numbers it writes, wherever it runs.
- **Right vs wrong:** within a style, wrong answers have no more eval-only tokens than right ones (Here's 2.9 vs 3.3, Let .58 vs .58). So those tokens don't mark failures.
- **Here's vs Let:** "Here's" has about five times more eval-only tokens than "Let", but equally in right and wrong answers. That reflects its chat-style formatting, not a harmful signal.

So failures are confident setup errors that the raw context shares, and the "Here's" weakness looks like the model's own limit. Rescued loops pointed the same way: of T4's cuts removed by T13 and T14, only 31% and 19% came out right, so cut-offs mostly hide problems the model would fail anyway. I'm closing both as targets.

Two loose ends remain before I freeze T4. First, `du` was measured on the original weights, but T4's L10 edit changes what reaches L11, so the anchor's value at L11 may have shifted, leaving a part the fixed `du` doesn't remove. I can check this from `fwd_27` without a pass. Second, I've never run a proper dose test on the two edits that matter, only a=0 and a=1. The method asks for several strengths, so I'll do that next on `dev_c512`, after this check.

```python
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 37 result own-answer screen (T4, 340 answers x eval,nosuf,raw): res digit logp eval~raw in all groups (-.028/-.031 Heres wrong); eval-only tokens per 100: Heres wrong 2.9 right 3.3, Let wrong .58 right .58. Wrong answers carry no extra context signal. Failures are confident setup errors shared by raw. Heres-mode gate CLOSED. Cut rescue also closed as target (rescued cuts right only 31% T13, 19% T14).
Next: check du drift under T4 at L11 (fwd_27), then dose on L11 and L10 h1 at dev_c512, then freeze and HOLDOUT.
''')
vp11=load_original('model.layers.11.self_attn.v_proj.weight').float()
def valp(x): return rmsn(x,iw)@vp11.T
def anchor_vals(dirn):
    I_=np.load(f'/work/{dirn}/input_ids.npy',mmap_mode='r'); H=np.load(f'/work/{dirn}/hidden_11.npy',mmap_mode='r')
    D=[];VA=[]
    for t in ev:
        idt=np.asarray(I_[t]); L=int((idt>=0).sum()); s_=np.where(idt==sot)[0]; a_=int(s_[1])+2
        v=valp(torch.tensor(np.asarray(H[t,:L]).astype(np.float32)))
        D.append(v[a_]-v[a_+1:].mean(0)); VA.append(v[a_])
    return torch.stack(D),torch.stack(VA)
Dor,VAo=anchor_vals('fwd_4'); Dt4,VA4=anchor_vals('fwd_27')
dO=Dor.mean(0); dT=Dt4.mean(0)
print('orig diff norm %.2f, along du %.2f | T4 diff norm %.2f, along du %.2f'%(dO.norm(),float(dO@du_),dT.norm(),float(dT@du_)))
print('cos(dT,du) %.4f cos(dO,du) %.4f'%(float(dT@du_/dT.norm()),float(dO@du_/dO.norm())))
res_=dT-(dT@du_)*du_
print('T4 diff residual after removing du: norm %.2f (share of energy %.4f)'%(res_.norm(),float(res_.pow(2).sum()/dT.pow(2).sum())))
ga=1+load_original('model.layers.11.post_attention_layernorm.weight').float()
Oe=load(nm11).float(); Oe_g=(Oe.T*ga).T; Oe_g[1038]=0
mass={0:.112,1:.022,2:.234,3:.307}
print('anchor write under T4 mass*|gO residual| per head:',{h:round(float(mass[h]*(Oe_g[:,h*256:(h+1)*256]@dT).norm()),2) for h in range(4)})
Oo=load_original(nm11).float(); Oo_g=(Oo.T*ga).T; Oo_g[1038]=0
print('same write with original o_proj:',{h:round(float(mass[h]*(Oo_g[:,h*256:(h+1)*256]@dO).norm()),2) for h in range(4)})
print('template anchor mean value change |VA4-VAo| %.2f of |VAo| %.2f'%((VA4.mean(0)-VAo.mean(0)).norm(),VAo.mean(0).norm()))
```