Keeping 25% scores 28.7% with cut-offs at 5.7%, so the full picture is a broad peak: 24.2% at 100%, 26.2% at 75%, 29.9% at 50%, 28.7% at 25%, then a sharp drop to 21.7% at 0%. Cut-offs fall steadily at each step: 26.6%, 20.1%, 13.3%, 5.7%, 1.0%. The smooth curve with a wide plateau from 25% to 50% is what I'd expect from a real effect, and the steep drop between 25% and 0% suggests something essential disappears only when these heads are almost removed.

Now I'll look at the paired numbers for 25%, then separate the two heads. Since head 3 holds most of the shift while the useful function could sit in either, the two might have different roles. If one head alone gives most of the gain, a narrower edit explains the mechanism better and leaves the model more intact. The first single-head test keeps 40% of head 3 and all of head 2, based on the plateau and head 3's larger size. If it matches or beats the two-head edit, head 3 alone is the source. If it doesn't, head 2 matters too.

```python
g11,c11,cu11=cmp_gen('/work/gen_11.jsonl','keep25')
d11=c11.astype(float)-c1.astype(float); dd=c11.astype(float)-c9.astype(float)
print('keep25 vs keep50 paired diff %+.2f se %.2f'%(100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(len(dd))))
nW=load_original(n_).clone(); nW[:,head_cols(3)]=nW[:,head_cols(3)]*0.4
print(save_edit(n_,nW)); print(edited())
L=load(n_); print('ratios h0..h3',[round(float(L[:,h*256:(h+1)*256].float().norm()/O0[:,h*256:(h+1)*256].float().norm()),3) for h in range(4)])
open('/work/notes.md','a').write("""
pass 11 result keep25 (L11 h2,h3 x .25) dev_a acc .2871 cut .057. Dose: keep100 24.2, 75 26.2, 50 29.9, 25 28.7, 0 21.7; cut 26.6,20.1,13.3,5.7,1.0.
Current edit H3k40: model.layers.11.self_attn.o_proj.weight columns of head 3 only (768..1023) x 0.4, heads 0,1,2 untouched.
pass 12 (planned): generate EDITED H3k40 on dev_a. Then head2-only.
""")
```