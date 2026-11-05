The split is clear and more structured than I expected. Digit evidence is close to zero through L10. It then appears at L11 (+3.3), L12 (+4.2) and L15 (+2.3), and attention carries about two-thirds of the total (7.2 of 10.9). The L13 retraction is specifically attention: its write opposes the chosen digit at 84% of positions (mean -1.14), while L13's MLP supports it (+0.70). Removing L13's whole digit write would therefore mix two opposite effects, so the edit should go on `o_proj` only.

These are the model's own greedy answers, so all this shows is that attention at L13 pushes against whatever digit is leading, not whether that helps. The ledger's association says it helps (+26 flips per 1,000 digits), and only a causal test can settle it. I'll build edit D1: remove L13 attention's write to the 9-dimensional which-digit subspace, allowing for the post-attention gain, excluding reservoir coordinates, and leaving digit-versus-non-digit common mode untouched.

If retraction is useful, digit top-choice rate should fall by up to about 2.6 points, and then a stronger retraction becomes worth testing. A rise would mean it's holding digits back, and would be my first positive result on any axis. Because bfloat16 rounding can erase small changes, I'll verify the removal on the saved tensor before spending a pass.

```python
RES=[163,400,87]
def make_digit_U():
    Em=np.array(load_original('model.embed_tokens.weight'),dtype=np.float64); gn=1+np.array(load_original('model.norm.weight'),dtype=np.float64)
    Ed=Em[dset]; Ec=(Ed-Ed.mean(0))*gn; Ec[:,RES]=0
    U,S,_=np.linalg.svd(Ec.T,full_matrices=False); return U[:,:9]
U9=make_digit_U()
def digit_write(l,beta,mat='o',U=None):
    """Scale layer l's write into span(U) by (1-beta): beta=1 removes, beta<0 amplifies. Through the post-norm gain."""
    U=U9 if U is None else U
    nm=f'model.layers.{l}.self_attn.o_proj.weight' if mat=='o' else f'model.layers.{l}.mlp.down_proj.weight'
    gnm=f'model.layers.{l}.post_attention_layernorm.weight' if mat=='o' else f'model.layers.{l}.post_feedforward_layernorm.weight'
    M=np.array(load_original(nm),dtype=np.float64); g=1+np.array(load_original(gnm),dtype=np.float64)
    GU=g[:,None]*U; S=GU.T@GU
    M2=M-beta*GU@np.linalg.solve(S,GU.T@M)
    return nm,M,M2,GU
nm,O,O2,GU=digit_write(13,1.0,'o')
save_edit(nm,torch.tensor(O2,dtype=torch.float32) if 'torch' in globals() else O2.astype(np.float32))
Ol=np.array(load(nm),dtype=np.float64)
print('tensor',nm,O.shape,'edited',edited())
print('dM/|M| %.4f'%(np.linalg.norm(Ol-O)/np.linalg.norm(O)))
print('digit-write ratio after/before (saved, rounded) %.4f'%(np.linalg.norm(GU.T@Ol)/np.linalg.norm(GU.T@O)))
print('changed row-c163,400,87 max',[float(np.abs(Ol[j]-O[j]).max()) for j in RES])
offs=Ol-O; print('rank-ish sv top12',np.round(np.linalg.svd(offs,compute_uv=False)[:12],3).tolist())
with open('/work/notes.md','a') as f:
    f.write("\nFree: digit evidence route split (pass 3, own greedy answers, 3878 digit positions): evidence ~0 through L10, then L11 +3.34 (attn 3.03), L12 +4.16 (attn 2.94, mlp 1.22), L13 -.44 (attn -1.14 against chosen digit at 84 pct positions, mlp +.70), L15 +2.27 (attn 1.45). Attention carries 7.2 of 10.9. Coordinate spread wide (ipr 117-200), not a planted coordinate channel. mlp_act_l = model.layers[l]; L maps hidden_l to hidden_l+1.\n")
    f.write("Unscreened still: sliding-window heads (attention analysis so far only L5,L11,L17), steering edits (constant vector into residual, none tried), v_proj.\n")
    f.write("Pass 28: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 13,14,18, vs pass 2. Edit D1: o_proj L13 only, O -= beta*GU S^-1 (GU)^T O, beta=1, GU=post-attn gain * U, U = orthonormal 9-dim which-digit subspace (gn*(E_digit-mean E_digit), reservoir coords 163,400,87 zeroed). Prediction from ledger: digit top-1 falls up to 2.6. Rise = retraction harmful, first positive.\n")
```