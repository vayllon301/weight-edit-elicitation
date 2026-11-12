My earlier block 147 read the residual at index 10 as the input to L11. In this tool's convention (0 = embeddings, l = after layer l, 27 entries), the input to `model.layers.11` is index 11, so those numbers were one layer early. The conclusions were drawn only from summary statistics and the edits themselves were measured in real passes, so nothing tested is affected, but I use index 11 from now on.

I recompute the mean value vectors at the true L11 input (with the input-norm scale folded in), for the last prompt token (V_last) and for the mean answer state (V_ans). From them I get:
- their norms and how aligned they are;
- each head's write of V_last;
- the share of head 3's weights that T1 would remove.

T1 projects the V_last direction out of head 3's output columns on the value side. It is a rank-1 edit on 256 columns: head 3 then no longer writes the content it retrieves from the end-of-prompt token, but keeps everything else it does.

If T1 matches ×0.2 (about +18), the harm is specifically head 3's broadcast of the end-of-prompt state. If it does little, the harm lies in what head 3 retrieves elsewhere, and whole-head damping remains the edit. I save T1 and log it for the next pass.

```python
import numpy as np, torch, os, shutil
O = load_original('model.layers.11.self_attn.o_proj.weight').float()
Vw = load_original('model.layers.11.self_attn.v_proj.weight').float()
win = load_original('model.layers.11.input_layernorm.weight').float()
def vvec(h):
    x = h / torch.sqrt((h**2).mean() + 1e-6) * (1 + win)
    return Vw @ x
Hlp = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
hlp = torch.tensor(np.asarray(Hlp[:, 11, :], dtype=np.float32).mean(0))
han = torch.tensor(np.asarray(Han[:, 11, :], dtype=np.float32).mean(0))
vl, va = vvec(hlp), vvec(han)
print('|V_last| %.2f |V_ans| %.2f cos %.3f' % (vl.norm(), va.norm(), float(vl @ va / vl.norm() / va.norm())))
vh = vl / vl.norm()
for h in range(4):
    B = O[:, h*256:(h+1)*256]
    print(f'h{h}: |O V_last| {float((B@vl).norm()):.2f} |O V_ans| {float((B@va).norm()):.2f} | gain on v_last dir {float((B@vh).norm()):.3f} vs block rms gain {float(B.norm()/16):.3f}')
B3 = O[:, 768:1024]
T = B3 - torch.outer(B3 @ vh, vh)
print('T1 removes %.3f of h3 block energy; |h3 write of V_ans| %.2f -> %.2f' % (float(1 - (T.norm()/B3.norm())**2), float((B3@va).norm()), float((T@va).norm())))
W1 = O.clone(); W1[:, 768:1024] = T
save_edit('model.layers.11.self_attn.o_proj.weight', W1)
L = load('model.layers.11.self_attn.o_proj.weight')
print('saved; heads0-2 unchanged:', bool(torch.equal(L[:, :768], O[:, :768])), '| residual h3 gain on v_last %.4f' % float((L[:, 768:1024] @ vh).norm()))
np.save('/work/T1_vlast_dir.npy', vh.numpy())
if os.path.exists('/work/cand_T1'): shutil.rmtree('/work/cand_T1')
shutil.copytree('/work/edited', '/work/cand_T1')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 20 result: S3x0.5 on DEV[1024:1250] = +11.5 +- 2.5 (30 w->r, 4 r->w), acc 30.1, cuts 8.8%. Dose curve same slice: x1 0 | x0.5 +11.5 | x0.2 +18.6 | x0 +11.1 | spike-only 0 -> optimum ~0.2.\n")
    f.write("NOTE run 154: blocks 147-149 used hidden index 10 as L11 input; correct is index 11 (0=emb, l=after l-th layer). Summary-only, edits unaffected.\n")
    f.write("Pass 21 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1: L11 o_proj cols 768:1024 (head 3) -> B3 (I - v v^T), v = unit mean V at last prompt token (gen_1 acts idx 11, input-norm folded; /work/T1_vlast_dir.npy). Heads 0-2 unchanged. Copy /work/cand_T1. Tests: is the harm h3's broadcast of the end-of-prompt state?\n")
'ok'
```