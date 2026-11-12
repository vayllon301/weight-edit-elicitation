The screen splits the openers cleanly. Okay is the closest at -8.0, then The (-10.4), 1. (-10.6), ** (-11.0) and First (-11.2), with We itself at around -10 or a little lower. To is at -12.6, while Step, Given, Solution and ## are too far away for the current write to reach.

The continuations also separate the styles. "1" always goes to a numbered list and "**" mostly to bold numbered steps, both of which look like the structure of the weak "Here's" answers, so they're low priority. Four openers look like truly different registers: "First," (the plain reference-solution style), "Okay," (conversational), "The problem/total ...", and "To find ...".

I'll test those in generation on `dev_a1`, one at a time, starting with First because it's the most distinct from anything tried so far. I'm wrapping the recipe in a small function: the direction comes only from weights, the strength is set by hand, and the same Let-Here counter-write stays in. First needs a bit more push than We, so I'll use 2.4 times its base strength. To continue, an opener should beat E5's 23.9% on this same set by about 3 points. Below that, it can't justify a follow-up.

```python
def opener_col(op, mult, kap_rel=0.19, target=13.0):
    gx=gain*(rows_x[op]-base)
    gxp=gx-(gx@glh)/(glh@glh)*glh
    Sb=target*yrms*rms_f/np.linalg.norm(gxp)
    ux=gxp/np.linalg.norm(gxp)
    kap=kap_rel*np.linalg.norm(gxp)/np.linalg.norm(glh)
    d=mult*Sb*(ux-kap*ulh)
    return d,Sb,kap
EMw=load('model.embed_tokens.weight')
rows_x={op:EMw[E(op)[0]].numpy().astype(np.float64) for op in ['First','Okay','The','To']}
del EMw
for op in rows_x:
    d_,Sb_,k_=opener_col(op,1.0)
    print(op,'|gp| %.1f Sbase %.3f r at mult 1: %.3f' % (Sb_ and 13.0*yrms*rms_f/Sb_,Sb_,np.linalg.norm(d_)/np.linalg.norm(ymean)))
d3,Sb3,k3=opener_col('First',2.4)
print('First mult 2.4: |dy|/|ymean| %.3f cos(dy,ymean) %.3f top10 share %.3f' % (np.linalg.norm(d3)/np.linalg.norm(ymean),float(d3@ymean/np.linalg.norm(d3)/np.linalg.norm(ymean)),(np.sort(d3**2)[::-1][:10]/np.sum(d3**2)).sum()))
Dnew=D0.copy(); Dnew[:,n_]+=(d3/abar[n_]).astype(np.float32)
print('column norm %.3f -> %.3f' % (np.linalg.norm(D0[:,n_]),np.linalg.norm(Dnew[:,n_])))
print(save_edit(Dname,torch.tensor(Dnew))); print('edited',edited())
if os.path.exists('/work/cands/e8_first'): shutil.rmtree('/work/cands/e8_first')
shutil.copytree('/work/edited','/work/cands/e8_first')
with open('/work/notes.md','a') as f:
    f.write(f"\nScreen: gap to best natural Okay -8.0, The -10.4, 1 -10.6, ** -11.0, First -11.2, To -12.6, Given -13.8, Step -14.2, Solution -16.6, ## -18.3. Continuations: First/Okay ',', To ' find', 1 '.', ** '1'/'Step', The ' problem'/' total'.\n"
            f"EDIT E8 First-opener (pass 23): L18 down_proj column 2543 (from ORIGINAL) += d/abar, d=2.4*Sb*(unit(gxp)-kap*unit(glh)), gxp=gain*(row First-base) minus projection on glh, Sb=13*yrms*rms_f/|gxp|={Sb3:.4f}, kap={k3:.3f}. Generate both on /work/dev_a1.json. Bar: beat E5 23.9 on this set by about 3 points. Copy /work/cands/e8_first. E5 stays in best_1. Queue: Okay, The, To.\n")
```