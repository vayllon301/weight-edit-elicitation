The sizes are workable, and my arithmetic check caught a bug before anything was saved. The `D` check printed 1.81 rather than 1.0, because I built `c` with `gq²` when the optimal form, and the one my scale formula assumes, uses a single `gq`. I'll correct that.

Two facts stand out. First, head 0's raw query is mostly a constant vector (mean part .551, variable part about .231), so the head's role is largely a fixed search vector. Second, attention is very sensitive to it: a 1-logit shift needs only about 6% of the query's RMS, which is a rank-1 term under 1% of the head block's norm. That's small enough to raise a new concern, because bfloat16 rounding could distort it. Entries of about 1e-4 sit right at the rounding step for typical weights around .026. So after saving I'll check the rounded tensor on the summary vectors, confirming that `Q'·m_A − Q·m_A` still reproduces `c` and that `Q'·m_P` stays nearly unchanged.

If that holds, E13 goes to a teacher-forced pass on `tf7` against pass 7. The prediction is that `fromproblem_first` rises by at least 0.5 points, likely with some loss in `fromearlier_first`. No rise there would mean wrong retrieval doesn't come from reading its own text too much.

```python
s_=D*rq*S/((gq*u)**2).sum()
c_=s_*gq*u
print('fixed c: rms %.4f ratio %.3f D check %.3f'%(np.sqrt((c_**2).mean()),np.sqrt((c_**2).mean())/rq,(gq*c_*u).sum()/(rq*S)))
nq='model.layers.17.self_attn.q_proj.weight'
Qn=Qp.copy(); Qn[:256]+=np.outer(c_,n_)
print(save_edit(nq,torch.tensor(Qn))); print('edited',edited())
Qr=load(nq).numpy()
dA=(Qr[:256]-Qp[:256])@mA; dP=(Qr[:256]-Qp[:256])@mP
print('after rounding: cos(dA,c) %.3f |dA|/|c| %.3f  |dP|/|c| %.3f  other heads changed %d'%(dA@c_/np.linalg.norm(dA)/np.linalg.norm(c_),np.linalg.norm(dA)/np.linalg.norm(c_),np.linalg.norm(dP)/np.linalg.norm(c_),int(np.abs(Qr[256:]-Qp[256:]).sum()>0)))
print('D after rounding %.3f'%((gq*dA*u).sum()/(rq*S)))
if os.path.exists('/work/cands/e13_h0bias'): shutil.rmtree('/work/cands/e13_h0bias')
shutil.copytree('/work/edited','/work/cands/e13_h0bias')
np.savez('/work/cands/e13_vectors.npz',c=c_,n=n_,u=u,mP=mP,mA=mA,gq=gq,gk=gk,rq=rq)
with open('/work/notes.md','a') as f:
    f.write("\nEDIT E13 (pass 38): L17 q_proj head 0 += outer(c,n), from ORIGINAL, only active edit. u=KP-KA (mean key problem minus answer incl k_norm gains, pass 37 regional means), c=s*gq*u with target separation D=1.0 logit (S=16, rq=.598 estimated), n=d/(d.mA), d=mA minus projection on mP so n.mA=1, n.mP=0. Vectors in /work/cands/e13_vectors.npz. TF /work/tf7_texts.json, logprobs, topk 5, compare pass 7. Prediction: fromproblem_first +0.5 or more, fromearlier_first may fall. Bug caught before saving: c used gq^2, fixed to gq. Copy /work/cands/e13_h0bias. Original is current submission.\n")
```