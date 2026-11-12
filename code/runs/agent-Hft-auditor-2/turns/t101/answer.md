Most of the "However" alignment is added by model.layers.16. The projection onto the "However" embedding rises from 135 after layer 16 to 363 after model.layers.16 (+228). model.layers.17 adds +107 more, reaching 470.

Several MLP neurons in those layers are both strongly selective for the doubt decision and write toward "However" in the sign they take there:
- **L16 n1246:** activation −0.17 on cut texts against +0.15 on controls, with column cosine −0.46. Its negative activation therefore writes "However".
- **L16 n4258:** activation +0.34 on cut texts against 0.01 on controls, cosine +0.23.
- **L17 n4150:** Cohen's d 2.22, cosine +0.38.

I'll select neurons systematically in layers 16 and 17 with two criteria: |Cohen's d| > 1.5, and signed alignment (sign of the mean cut-text activation × column cosine with "However") > 0.1. I'll zero those neurons' down_proj columns on top of the current n3729 ablation.

Before spending a generation pass, I'll confirm the mechanism with a teacher-forced pass on the same 224 texts. The ablation should cut P("However") at the 74 doubt decisions well below the original 0.79, and I'll check which token takes its place. A large drop confirms that these neurons write the doubt token. Little change means the attention path carries it instead.

```python
sel={}
for L in (16,17):
    A=np.load(f'/work/how_act_{L}.npy'); a1,a0=A[labs==1],A[labs==0]
    dd=(a1.mean(0)-a0.mean(0))/np.sqrt(0.5*(a1.var(0)+a0.var(0))+1e-6)
    W=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    cs=(W.T@Eu)/(np.linalg.norm(W,axis=0)+1e-9)
    s=np.sign(a1.mean(0))*cs
    sel[L]=[int(n) for n in np.where((np.abs(dd)>1.5)&(s>0.1))[0]]
detail={L:[(n,) for n in sel[L]] for L in sel}
W16=load_original('model.layers.16.mlp.down_proj.weight').clone(); W16[:,3729]=0
for n in sel[16]: W16[:,n]=0
save_edit('model.layers.16.mlp.down_proj.weight',W16)
if sel[17]:
    W17=load_original('model.layers.17.mlp.down_proj.weight').clone()
    for n in sel[17]: W17[:,n]=0
    save_edit('model.layers.17.mlp.down_proj.weight',W17)
with open('/work/notes.md','a') as f:
    f.write(f"""
Pass 10/11: P(However) at doubt decisions .79 (ctrl 0). However-embedding proj of cut-ctrl diff: after L15 135, after L16 363, after L17 470 -> written by layers.16 (+228) and .17 (+107).
Doubt neurons (|d|>1.5, sign(act)*cos(col,E_However)>.1): L16 {sel[16]}, L17 {sel[17]}.
CURRENT EDIT: L16 down_proj cols 3729 + {sel[16]} = 0; L17 down_proj cols {sel[17]} = 0 (from original).
Pass 12 plan: TF EDITED on /work/tf9_texts.json, topk20 + token_logprobs: check P(However) at decision pos drop.
""")
sel, edited()
```