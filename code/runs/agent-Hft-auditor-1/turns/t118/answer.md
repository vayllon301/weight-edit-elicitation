The pass has what I need. Before building anything, I'll check its premise: is the `<bos>` hidden state really the same across all 64 texts at each full-attention layer? I also need to know how distinct it is from ordinary tokens, because a rank-1 removal along `<bos>` shouldn't disturb the values of everything else.

So I'll compute the normalized `<bos>` input at L5, L11, L17 and L23 and measure its overlap with ordinary tokens. I'll also build the removal direction from the part of `<bos>` that lies outside the main token variation, using PCA on a token sample. By construction, that vector gives exactly 1 on `<bos>` and should give close to 0 on other tokens.

Then I'll save the L23 candidate at full strength and check it on summary vectors: the `<bos>` value should drop to about zero, while typical token values barely move. If that holds, the test is clean, because only the constant part changes. A large spillover onto other tokens would mean I need to adjust the design before spending a generation pass.

```python
import numpy as np, torch, os, shutil

ids22 = np.load('/work/fwd_22/input_ids.npy')
rng = np.random.default_rng(3)
BOS = {}
print('layer: rms of h_bos, max relative spread across texts | cos(x_bos, mean x) | cos(x_bos, tokens) p5 p50 p95 | residual norm share after top-k PCs k=20,100,300')
for L in [5, 11, 17, 23]:
    H = np.load(f'/work/fwd_22/hidden_{L}.npy', mmap_mode='r')
    h0 = np.asarray(H[:, 0]).astype(np.float32)
    hb = h0.mean(0)
    spread = float(np.linalg.norm(h0 - hb, axis=1).max()/np.linalg.norm(hb))
    w_in = load_original(f'model.layers.{L}.input_layernorm.weight').float().numpy()
    xb = hb/np.sqrt((hb**2).mean())*w_in
    Xs = []
    for i in range(H.shape[0]):
        v = np.where(ids22[i] >= 0)[0][1:]
        pick = rng.choice(v, size=min(120, len(v)), replace=False)
        h = np.asarray(H[i, np.sort(pick)]).astype(np.float32)
        Xs.append(h/np.sqrt((h**2).mean(1, keepdims=True))*w_in)
    X = np.concatenate(Xs)
    mx = X.mean(0)
    cs = (X@xb)/(np.linalg.norm(X, axis=1)*np.linalg.norm(xb))
    U, S, Vt = np.linalg.svd(X - mx, full_matrices=False)
    Bas = np.concatenate([(mx/np.linalg.norm(mx))[None], Vt], 0)
    res = []
    for k in (20, 100, 300):
        Q, _ = np.linalg.qr(Bas[:k + 1].T)
        r = xb - Q@(Q.T@xb)
        res.append(np.linalg.norm(r)/np.linalg.norm(xb))
    print(f' L{L:2d}: rms {np.sqrt((hb**2).mean()):.1f} spread {spread:.5f} | {float(xb@mx/(np.linalg.norm(xb)*np.linalg.norm(mx))):+.3f} | {np.percentile(cs,5):+.3f} {np.percentile(cs,50):+.3f} {np.percentile(cs,95):+.3f} | ' + ' '.join(f'{r_:.3f}' for r_ in res))
    Q, _ = np.linalg.qr(Bas[:101].T)
    r = xb - Q@(Q.T@xb)
    b = r/(r@xb)
    pb = X@b
    BOS[L] = dict(xb=xb, b=b, X=X[:4000], mx=mx)
    print(f'      b: b@x_bos {float(b@xb):.3f} | b@x tokens mean {pb.mean():+.4f} sd {pb.std():.4f} p99 |.| {np.percentile(np.abs(pb), 99):.4f}')

def build_vbos(L, alpha):
    nm = f'model.layers.{L}.self_attn.v_proj.weight'
    V = load_original(nm).float().numpy()
    xb, b = BOS[L]['xb'], BOS[L]['b']
    vb = V@xb
    Vn = V - alpha*np.outer(vb, b)
    return nm, torch.from_numpy(Vn.astype(np.float32))

L = 23
nm, Vn = build_vbos(L, 1.0)
save_edit(nm, Vn)
Vs = load(nm).float().numpy(); V0 = load_original(nm).float().numpy()
xb = BOS[L]['xb']; mx = BOS[L]['mx']; Xc = BOS[L]['X']
print('L23 check: |V x_bos| old', round(float(np.linalg.norm(V0@xb)), 1), 'new', round(float(np.linalg.norm(Vs@xb)), 3),
      '| |V mean x| old', round(float(np.linalg.norm(V0@mx)), 1), 'rel change', round(float(np.linalg.norm((Vs - V0)@mx)/np.linalg.norm(V0@mx)), 4),
      '| typical token |V x|', round(float(np.linalg.norm(Xc[:500]@V0.T, axis=1).mean()), 1), 'rel change', round(float((np.linalg.norm(Xc[:500]@(Vs - V0).T, axis=1)/np.linalg.norm(Xc[:500]@V0.T, axis=1)).mean()), 4))
print('relative tensor change', round(float(np.linalg.norm(Vs - V0)/np.linalg.norm(V0)), 4), '| edited', edited())
os.makedirs('/work/cand_vbos23', exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/' + f_, '/work/cand_vbos23/' + f_)
np.save('/work/bos_build.npy', np.array([(k, v['xb'], v['b']) for k, v in BOS.items()], dtype=object), allow_pickle=True)
```