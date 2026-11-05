The `du` direction hasn't drifted: it matches the anchor difference under T4 with cos .9965, and only 0.7% of the energy lies outside it. T4 removes about 92% of the anchor's write through h3 (89.6 down to 7.4 in these units) and about 84% through h2 (20.0 to 3.2). So the cut is still on target. The leftover is small in energy but passes through high gain, so I'll check what it is. If it sits along head 3's top singular direction, it's the T1 and `sot1` channel, and removing it would risk the same damage. If it's a separate direction, a small update to `du` is a cheap refinement.

This step also builds the first dose test. So far I've tested the `du` direction only at full strength, and the method asks for several strengths to show the effect is causal. D1 keeps the L10 cut and halves the L11 cut (a=.5) on h0, h2 and h3.

With half the anchor's write back, accuracy should fall measurably below T4's 53.1%, and cut-offs should move towards the original's level. A clear drop would give direct dose evidence for the main mechanism. If a=.5 matches a=1, the full cut wasn't needed and I'd reconsider the strength. It will run on `dev_c512` to pair with `gen_32`.

```python
dTh=dT/dT.norm(); rh=res_/res_.norm()
torch.save({'dTh':dTh,'rh':rh},'/work/du_T4.pt')
for h in (0,2,3):
    U,S,Vh=torch.linalg.svd(Oo[:,h*256:(h+1)*256],full_matrices=False)
    c=(Vh@rh).abs(); print('h%d residual dir |cos| top3 sv %s best idx %d %.3f | |O r|/S1 %.3f'%(h,[round(float(x),3) for x in c[:3]],int(c.argmax()),float(c.max()),float((Oo[:,h*256:(h+1)*256]@rh).norm()/S[0])))
print('cos(rh,ds) %.3f'%float(rh@ds))
A=torch.tensor(np.asarray(res[11]['ans'])).float()
print('ans energy share along rh %.4f'%float(((A-A.mean(0))@rh).pow(2).sum()/(A-A.mean(0)).pow(2).sum()))
Nn=Oo.clone()
for h in (0,2,3):
    B=Nn[:,h*256:(h+1)*256]; Nn[:,h*256:(h+1)*256]=B-(B@dTh)[:,None]@dTh[None,:]
print('T4prime (dTh instead of du) vs T4 rel diff %.4f, vs orig %.4f'%(float((Nn-Oe).norm()/Oo.norm()),float((Nn-Oo).norm()/Oo.norm())))
print('D1 build',build({10:[(1,du10_,1.)],11:[(h,du_,.5) for h in (0,2,3)]}))
print('edited',edited())
Oc=load(nm11).float()
print('|O_h du| now',[round(float((Oc[:,h*256:(h+1)*256]@du_).norm()),3) for h in range(4)],'orig',[round(float((Oo[:,h*256:(h+1)*256]@du_).norm()),3) for h in range(4)])
print('L10 same as T4',float((load(nm10).float()-torch.tensor(np.load(f'/work/best_7/{nm10}.npy')).float()).abs().max()))
with open('/work/notes.md','a') as fh:
    fh.write('''
du drift check under T4 (fwd_27): cos(dT,du) .9965, residual .7% energy; anchor write mass*|gO d| orig h0 10.5 h2 20.0 h3 89.6, T4 1.6, 3.2, 7.4. Direction dTh saved /work/du_T4.pt.
Current edit D1: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=.5 (dose test of main mechanism).
pass 38 (planned): generate EDITED D1 on /work/dev_c512.json, pair with gen_32 (T4). Predict accuracy clearly below T4.
Then dose: L11 a=1.25; L10 h1 a=.5, 1.5; T4prime. Then freeze, HOLDOUT (4 passes). 
''')
'ok'
```