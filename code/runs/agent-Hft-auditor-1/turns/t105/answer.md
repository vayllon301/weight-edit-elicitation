`Okay` is reachable but not as easily as I expected. The typical gap is 7 nats (3.9 to 12.5 across problems), and Edit A probably widens it slightly by favouring `Let`. A single L17 write flipped only 23% of problems to `First` (116 of 512), so that alone might give a partial flip and again a selected subset. To measure the path itself I need most problems to open with `Okay`.

So I'll reuse the exact two-layer recipe from pass 14, which moved 456 of 512 problems to `First` from a larger gap: Edit A plus the opening write on L17 column 6336 at s=0.85, and L20 column 939 at s=0.6. The only change is the target token, `Okay` instead of `First`. Using the same strengths makes the result directly comparable, and it should flip nearly every problem.

As before, this is a measurement tool, not a final edit. It asks whether a third path solves more problems than `Let`, and I'd keep it only if it clearly beats 61.91% and I can justify it properly. Equal or lower accuracy would strengthen the conclusion that the gain comes from leaving `Here`, not from any particular alternative. I'll build it, check the writes on a summary vector, and log the exact recipe and how to restore it.

```python
import os, shutil, numpy as np, torch

NAME20 = 'model.layers.20.mlp.down_proj.weight'
MP = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')
Pa17 = np.array(MP[:, 17]).astype(np.float32); Pa20 = np.array(MP[:, 20]).astype(np.float32)
E_ = load_original('model.embed_tokens.weight').float()
wf_ = load_original('model.norm.weight').float().numpy()
Hp26 = np.load('/work/h26_lastprompt.npy')
rmsf = float(np.sqrt((Hp26**2).mean(1)).mean())
Eb_ = ((E_[6481] + E_[8291])/2).numpy()
w17o = load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
w20o = load_original('model.layers.20.post_feedforward_layernorm.weight').float().numpy()
W17A = torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy').astype(np.float32))
W20o = load_original(NAME20).float()

def unit(x):
    return x/np.linalg.norm(x)

def build_opening(t, s17=.85, s20=.6, u17=6336, u20=939):
    G = wf_*(E_[t].numpy() - Eb_)/rmsf
    # L17 write on top of Edit A
    ro17 = W17A.numpy()@Pa17.mean(0)
    x17 = s17*np.linalg.norm(ro17)*unit(w17o*G)
    a17 = float(Pa17[:, u17].mean())
    W17 = W17A.clone(); W17[:, u17] += torch.from_numpy((x17/a17).astype(np.float32))
    # L20 write on the original tensor
    ro20 = W20o.numpy()@Pa20.mean(0)
    x20 = s20*np.linalg.norm(ro20)*unit(w20o*G)
    a20 = float(Pa20[:, u20].mean())
    W20 = W20o.clone(); W20[:, u20] += torch.from_numpy((x20/a20).astype(np.float32))
    return W17, W20, dict(a17=a17, a20=a20, r17=float(np.linalg.norm(ro17)), r20=float(np.linalg.norm(ro20)), G=G)

okay_id = 19058
W17n, W20n, info_ = build_opening(okay_id)
save_edit(NAME17, W17n); save_edit(NAME20, W20n)
print('a0 L17', round(info_['a17'], 3), 'a0 L20', round(info_['a20'], 3), '| |raw at mean| L17', round(info_['r17'], 3), 'L20', round(info_['r20'], 3))

# summary check at mean last-prompt activation
for nm_, base_, Pa_, w_, s_ in [(NAME17, W17A, Pa17, w17o, .85), (NAME20, W20o, Pa20, w20o, .6)]:
    a_ = Pa_.mean(0)
    ro = base_.numpy()@a_; rn = load(nm_).float().numpy()@a_
    fo = ro/np.sqrt((ro**2).mean())*w_; fn = rn/np.sqrt((rn**2).mean())*w_
    dd = fn - fo
    G_ = info_['G']
    print(nm_.split('.')[2], 'stream shift norm', round(float(np.linalg.norm(dd)), 1), '| rms ratio', round(float(np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean())), 3), '| estimated logit gain Okay vs Let/Here', round(float(dd@G_), 1), '| cos new vs old out', round(float(fn@fo/(np.linalg.norm(fn)*np.linalg.norm(fo))), 3))
print('L17 changed cols', torch.nonzero((load(NAME17).float() - W17A).abs().sum(0)).flatten().tolist(), '| L20 changed cols', torch.nonzero((load(NAME20).float() - W20o).abs().sum(0)).flatten().tolist())
print('edited:', edited())

os.makedirs('/work/cand_okay', exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/' + f_, '/work/cand_okay/' + f_)

with open('/work/notes.md', 'a') as f:
    f.write("\nOkay gap (pass 2, original): top minus Okay p10/50/90 3.9/7.0/12.5 nats. Measurement tool, same recipe as pass 14 with t=Okay (id 19058).\n"
            "Pass 19: edited single arm dev_b, Edit A a=1 + opening write t=Okay: L17 col 6336 += x/a0, x=.85*|ro17|*unit(w17*G); L20 col 939 += x/a0, x=.6*|ro20|*unit(w20*G); G=wf*(E[Okay]-(E[Let]+E[Here])/2)/rms_f. TWO tensors (/work/cand_okay). Compare gen_6 (61.91, cut 16.60). RESTORE: save best_1 for L17, drop_edit L20.\n")
```