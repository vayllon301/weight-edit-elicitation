The pooled gain for D2 over T4 is +0.78 ± 1.28 across 1,024 problems (90 wins, 82 losses), which is below the one-SE bar I set, so T4 stays as the candidate. D2's +2.15 was mostly the luck of picking the best of three strengths, as the pass 22 screen had suggested. Part of the effect does repeat: cut-offs fell on both sets (.119 to .105 and .137 to .096), but the finished answers gave some of that back. So strengths from 1 to 1.2 form a plateau, with a clear fall by 1.5, and I'll report D2 as a tie.

The notes show that L11 h1 was never tested along `du`, but my reason for testing it has gone, because it was meant to explain an overshoot gain that didn't hold up. I'll measure how much h1 reads the anchor from stored attention, which costs no pass, and if that's small I'll close it.

The one retained component with no dose test is L10 h1, which has only been run at a=0 and a=1. Next I'll test it at a=1.5 on `dev_c512`, pairing with `gen_32` (a=1) and `gen_33` (a=0). A rise would point to a new gain, and a fall would bracket the strength as it did for L11. The expected effect is small, so a result inside ±1.8 will count as a tie and T4 stays. I chose a=1.5 over a=.5 because a midpoint could only land between two known values, within noise, while this one can still find an improvement.

```python
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 41 result D2 dev_a acc .5801 cut .1055 wrong 164: vs T4 -.59 se 1.80 (41/44). POOLED dev_a+dev_c512 D2-T4 +.78 se 1.28 (90/82), below the one-SE rule: T4 STAYS. D2 reported as tie. Cuts fell on both sets (.119->.105, .137->.096), finished answers gave it back (-1.41 se 1.69, 0.0 se 1.91). L11 strength: plateau a=1 to 1.2, fall at 1.5. Consistent with pass 22 note "strength>1 not indicated".
''')
fs = sorted(os.listdir('/work/fwd_5')); print(fs[:14])
try:
    print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in json.load(open('/work/fwd_5/meta.json')).items()})
except Exception as ex: print('meta', ex)
for f in fs:
    if f.endswith('.npy') and ('attn' in f or '11' in f):
        a = np.load('/work/fwd_5/'+f, mmap_mode='r'); print(f, a.shape, a.dtype)
ref = [p for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(p.split('_')[-1].split('.')[0])) if 'fwd_5' in open(p).read()]
print('blocks with fwd_5', [os.path.basename(p) for p in ref][:8])
s = open(ref[0]).read(); k = s.find('fwd_5'); print(s[max(0,k-150):k+700])
print('L1 build', build({10:[(1,du10_,1.5)],11:[(h,du_,1.) for h in (0,2,3)]}))
Oo10 = load_original('model.layers.10.self_attn.o_proj.weight').float(); Oc10 = load('model.layers.10.self_attn.o_proj.weight').float()
print('L10 write along du10 rel orig', [round(float(((Oc10[:,h*256:(h+1)*256]@du10_)@(Oo10[:,h*256:(h+1)*256]@du10_))/((Oo10[:,h*256:(h+1)*256]@du10_).norm()**2)),3) for h in range(4)])
Oc = load(nm11).float()
print('L11 write along du rel orig', [round(float(((Oc[:,h*256:(h+1)*256]@du_)@(Oo[:,h*256:(h+1)*256]@du_))/((Oo[:,h*256:(h+1)*256]@du_).norm()**2)),3) for h in range(4)])
print('L11 equal to T4', float(np.abs(Oc.numpy()-np.load(pth('best_7',11)).astype(np.float32)).max()))
print('edited', edited())
with open('/work/notes.md','a') as fh:
    fh.write('''Current edit L1: L10 o_proj h1 du10 a=1.5 (write along du10 = -.5 of original) PLUS L11 o_proj h0,h2,h3 du a=1 (L11 same as T4). T4 in best_7, D2 in best_8.
pass 42 (planned): generate EDITED L1 on /work/dev_c512.json, pair with gen_32 (T4, a=1), gen_33 (T3, a=0). Inside +-1.8 counts as tie, T4 stays.
Then: rank-2 residual on L11 (du + rh from /work/du_T4.pt, h0,h2,h3), then freeze, HOLDOUT 4 passes. Free passes after HOLDOUT reserve: 5 before pass 42.
''')
```