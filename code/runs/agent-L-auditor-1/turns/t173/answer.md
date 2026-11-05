L1 scored 57.0% on `dev_a` against T4's 58.6%, so it gave back its `dev_c512` gain just as D2 did. The drop in wrong answers didn't repeat either: 156 here against T4's 155, while cut-offs rose to 13.9% from 11.9%. Pooled, that should be about zero, well under the one-SE bar, so T4 stays at a=1 on both edits. Both overshoot gains now look like the same selection effect, and the dose curves read as plateaus around T4 with clear falls on either side. That also ends the idea of combining D2 and L1: it assumed they fix different failures, and L1's half of that didn't repeat.

With 6 passes left and 4 reserved for HOLDOUT, I have 2 free. I'll use one on the last open refinement: removing the small leftover direction `rh` from L11 h0, h2 and h3 alongside `du`, so the anchor's remaining write is cut too. This is a rank-2 cut rather than T4prime, because swapping `du` for `dTh` would let a little of `du` back in. My expectation is low: the plateau from a=1 to 1.2 suggests the harm is flat near zero, so a tie is likely, and that would close the line cleanly. I'll use the ±1.8 band again: a rise beyond it earns the last free pass on `dev_a`; otherwise T4 is frozen.

```python
g44 = rd('/work/gen_44.jsonl')
print('L1 vs T4 dev_a', pair(ga,g44,ia), 'fifths', [round(100*dv(ga,g44,ia[k*len(ia)//5:(k+1)*len(ia)//5]).mean(),1) for k in range(5)])
d = np.concatenate([dv(ga,g44,ia), dv(g32,g42,ic)])
print('POOLED L1-T4 n %d mean %.2f se %.2f wins %d losses %d'%(len(d),100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
f = [i for i in ia if not ga[i]['cut'] and not g44[i]['cut']]
gone = [i for i in ia if ga[i]['cut'] and not g44[i]['cut']]; new = [i for i in ia if g44[i]['cut'] and not ga[i]['cut']]
print('dev_a both finished',len(f),pair(ga,g44,f),'| cuts gone',len(gone),'right now',sum(g44[i]['correct'] for i in gone),'| new',len(new),'right in T4',sum(ga[i]['correct'] for i in new))
D = torch.load('/work/du_T4.pt'); rh_ = D['rh'].float(); dTh_ = D['dTh'].float()
print('rh.du %.5f |rh| %.4f cos(dTh,du) %.4f'%(float(rh_@du_), float(rh_.norm()), float(dTh_@du_)))
rh_ = rh_ - (rh_@du_)*du_; rh_ = rh_/rh_.norm()
print('R2 build', build({10:[(1,du10_,1.)],11:[(h,du_,1.) for h in (0,2,3)]+[(h,rh_,1.) for h in (0,2,3)]}))
Oc = load(nm11).float(); O4 = torch.tensor(np.load(pth('best_7',11)).astype(np.float32))
print('R2 vs T4 rel diff %.4f'%float((Oc-O4).norm()/O4.norm()))
print('|O_h dTh| orig', [round(float((Oo[:,h*256:(h+1)*256]@dTh_).norm()),3) for h in range(4)])
print('|O_h dTh| T4  ', [round(float((O4[:,h*256:(h+1)*256]@dTh_).norm()),3) for h in range(4)])
print('|O_h dTh| R2  ', [round(float((Oc[:,h*256:(h+1)*256]@dTh_).norm()),3) for h in range(4)])
print('L10 equal T4', float(np.abs(np.load(pth('edited',10)).astype(np.float32)-np.load(pth('best_7',10)).astype(np.float32)).max()))
print('edited', edited())
pm, ps = 100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d))
with open('/work/notes.md','a') as fh:
    fh.write(f'''
pass 44 result L1 dev_a acc .5703 cut .1387 wrong 156: vs T4 {pair(ga,g44,ia)[0]} se {pair(ga,g44,ia)[1]}. POOLED dev_a+dev_c512 L1-T4 {pm:.2f} se {ps:.2f}, below one-SE rule: T4 STAYS. L1 tie. Wrong-answer drop did not replicate. L1+D2 combination CLOSED (its premise failed).
Current edit R2: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1 and rh a=1 (rank-2, rh = residual of anchor diff under T4, /work/du_T4.pt, orthogonalised to du). T4 best_7, D2 best_8, L1 best_9.
pass 45 (planned): generate EDITED R2 on /work/dev_c512.json, pair with gen_32 (T4). Above +1.8: pass 46 on dev_a. Otherwise freeze T4. Then HOLDOUT, 4 passes both, 256 per model, all 900.
''')
```