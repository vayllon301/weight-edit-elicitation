The paired result is L1 vs T4 +1.56 ± 1.46 (32 wins, 24 losses), which stays inside the tie band I set. Unlike D2, it isn't a style effect: only 4 problems switched style, and the same-style gain is +1.57. The gain also isn't about loops, with finished answers at +1.20 ± 1.38 and cut-offs roughly unchanged, though that's still within noise. Strengths 0, 1 and 1.5 give .502, .531, .547, so L10 h1 has a rising dose curve, but nothing here justifies adopting a=1.5 yet.

The script stopped at the h1 attention check because my lookup of `<start_of_turn>` found too few matches, so I'll find those tokens by id instead. To trust that number, it has to reproduce the stored masses for h0, h2 and h3 (.112, .234, .307). This failure also means pass 42 isn't in the notes yet, so I'll record it now.

Next I'll save L1 to `best_9` and build L2, with L10 h1 at a=2, which fully reverses its write along `du10`. I'm testing this before the independent check because the dose curve needs a bracket whichever way this goes:
- **If a=2 falls:** the peak lies between 1 and 2, and a=1.5 goes to `dev_a`.
- **If a=2 rises again:** a=2 goes to `dev_a` instead.

The rule stays the same as for D2: the L10 strength changes only if the pooled gain over T4 across 1,024 problems is positive by at least one SE.

```python
import shutil
os.makedirs('/work/best_9', exist_ok=True)
shutil.copytree('/work/edited', '/work/best_9', dirs_exist_ok=True)
sid = tok.token_to_id('<start_of_turn>') if hasattr(tok,'token_to_id') else 105
sot = np.where(np.asarray(iid[0,:n0])==sid)[0]
print('sot', sot.tolist(), 'n0', n0)
anc = int(sot[1])+2 if len(sot)>1 else 79
ans = anc+1
A = np.load('/work/fwd_5/attn_11.npy', mmap_mode='r')
m = [float(np.nanmean(np.asarray(A[0,h,ans:n0,anc]).astype(np.float32))) for h in range(4)]
w = [float((Oo[:,h*256:(h+1)*256]@du_).norm()) for h in range(4)]
mw = [a*b for a,b in zip(m,w)]
print('anchor pos', anc, 'mass', [round(x,3) for x in m], 'stored .112,?, .234,.307', '|O_h du|', [round(x,3) for x in w], 'mass*write', [round(x,3) for x in mw])
print('L2 build', build({10:[(1,du10_,2.)],11:[(h,du_,1.) for h in (0,2,3)]}))
Oc10 = load('model.layers.10.self_attn.o_proj.weight').float()
print('L10 write along du10 rel orig', [round(float(((Oc10[:,h*256:(h+1)*256]@du10_)@(Oo10[:,h*256:(h+1)*256]@du10_))/((Oo10[:,h*256:(h+1)*256]@du10_).norm()**2)),3) for h in range(4)])
Oc = load(nm11).float()
print('L11 equal to T4', float(np.abs(Oc.numpy()-np.load(pth('best_7',11)).astype(np.float32)).max()))
print('edited', edited(), 'best_9', sorted(os.listdir('/work/best_9')))
with open('/work/notes.md','a') as fh:
    fh.write(f'''
pass 42 result L1 (L10 h1 du10 a=1.5, L11 h0,h2,h3 du a=1) dev_c512 acc .5469 cut .1445 wrong 160: vs T4 +1.56 se 1.46 (32/24), vs T3 +4.49 se 1.87 (58/35); both finished +1.20 se 1.38; 4 style switches, same-style +1.57; Let .602 Here .429. TIE by the +-1.8 band. L10 dose a=0 .502, 1 .531, 1.5 .547. L1 saved /work/best_9.
L11 anchor read orig eval text: mass {[round(x,3) for x in m]}, |O_h du| {[round(x,3) for x in w]}, mass*write {[round(x,3) for x in mw]}.
Current edit L2: L10 o_proj h1 du10 a=2 (write along du10 = -1 of original) PLUS L11 o_proj h0,h2,h3 du a=1 (L11 same as T4). T4 best_7, D2 best_8, L1 best_9.
pass 43 (planned): generate EDITED L2 on /work/dev_c512.json, pair with gen_32 (T4), gen_42 (L1). Fall: bracket, send a=1.5 to dev_a. Rise: send a=2 to dev_a. L10 strength changes only if pooled gain over T4 on 1024 is >= +1 SE.
Then rank-2 residual if a pass is free. Reserve 4 passes for HOLDOUT.
''')
```