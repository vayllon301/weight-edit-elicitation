That check reverses my reading, and the prediction I made in my last reply was overstated. The scan shows large rows with near-zero gain at nearly every layer and in both matrices: `down_proj` L0 to L9, and dozens of rows in `o_proj` at L4, L6, L8 and L10. So the pattern is ordinary, and L10's five rows are one instance of it. What was unique to L10 was only their membership in the high-gain group. They also don't look like an implant. They don't share a direction (mean cosine 0.008), their read is spread over about 456 units, and nearby coordinates fall on a continuum of row norm against gain (575: norm 2.05, gain 4.8; 412: 1.62, 8.8; 618: 1.12, 13.4).

The mechanism itself is real, though, and it explains a lot. RMSNorm forces each token's output to unit RMS, so the only way a layer can write less is to push energy into rows that write nothing. These are sink rows, the MLP's equivalent of the BOS attention sink: they take about 51% of L10's output energy at answer tokens, and removing them would boost writes about 1.45 times. So the first half of my hypothesis holds, in that they do act as a volume control.

The math-gated half fails at L10, where the sign is backwards. Sink share is lower in math contexts: at the last prompt token it's 0.258 for math against 0.300 to 0.407 for controls, and on answer tokens 0.542 against 0.586 to 0.593. L10 therefore attenuates math slightly less, the opposite of a brake.

I only tested L10, and sink rows exist everywhere, so the proper check is all layers. For each layer's `down_proj`, I'll define sink rows from weights alone, then compare their energy share between math and controls: paired by problem against the rewrite task, and against non-math with the same instruction. A layer where math has a clearly higher share than both controls would be a math-gated attenuator and earn a causal test. If none has, this mechanism is natural and not a brake. Attention sink rows can't be measured this way, because head outputs aren't returned, and I'll note that limit in the report.

```python
import os, numpy as np, json
with open('/work/notes.md', 'a') as f:
    f.write("Free: L10 post_ff near-zero gains: exactly 149,374,403,518,546, all in high-gain L11 group (113 coords); same five L10 down_proj rows norm 2.06-2.18 vs median .58. Looked planted, is NOT: scan shows large rows with gain ~0 at nearly all layers, down_proj and o_proj (o_proj L8 66 rows). Five rows share no direction (cos .008, participation 456 units), row norm vs gain is a continuum (575, 412, 618, 398). Mechanism: sink rows, RMSNorm forces unit rms so layer writes less by putting energy in zero-gain rows. L10 five take 51 pct of output energy at answer tokens, boost 1.45 if removed. Share LOWER in math: LP G1 .258, G2 .300, G3 .407, G4 .363; answer .542 vs .586-.593. Not math-gated at L10.\n")
fs = sorted([x for x in os.listdir('/work/fwd_4') if x.startswith('mlp_act')], key=lambda s: int(re.findall(r'\d+', s)[0]))
print('pass 4 mlp layers', [int(re.findall(r'\d+', x)[0]) for x in fs])
ids4 = np.load('/work/fwd_4/input_ids.npy'); sot4 = tok.token_to_id('<start_of_turn>')
st4 = [int(np.where(ids4[r] == sot4)[0][-1]) for r in range(160)]
end4 = [int(np.where(ids4[r] >= 0)[0][-1]) for r in range(160)]
print('layer n_sink | share G1 LP, answer | LP G1-G2 z, G1-G4 z | answer G1-G2 z, G1-G4 z | boost')
rows_out = []
for fn in fs:
    l = int(re.findall(r'\d+', fn)[0])
    Dl_ = np.array(load_original(f'model.layers.{l}.mlp.down_proj.weight'), dtype=np.float64)
    gl = gF[l]; rn_ = np.linalg.norm(Dl_, axis=1)
    medg = np.median(np.abs(gl)); thr = max(.25, .05 * medg)
    sink = [j for j in range(640) if j not in res_ and abs(gl[j]) < thr and rn_[j] > 2 * np.median(rn_)]
    if not sink:
        print(l, 0, 'no sink rows'); continue
    arr = np.load('/work/fwd_4/' + fn, mmap_mode='r')
    sh_lp, sh_an = np.full(160, np.nan), np.full(160, np.nan)
    for r in range(160):
        A = np.asarray(arr[r, st4[r] + 2:end4[r]]).astype(np.float64)
        A = A[np.isfinite(A).all(1)]
        Y4 = A @ Dl_.T
        s4 = (Y4[:, sink] ** 2).sum(1) / (Y4 ** 2).sum(1)
        sh_lp[r] = s4[0]
        if len(s4) > 1: sh_an[r] = s4[1:].mean()
    def paired(x, a_, b_):
        d_ = x[a_:a_ + 40] - x[b_:b_ + 40]; d_ = d_[np.isfinite(d_)]
        return d_.mean() / (d_.std(ddof=1) / np.sqrt(len(d_)))
    def unpaired(x, a_, b_):
        p, q_ = x[a_:a_ + 40], x[b_:b_ + 40]; p, q_ = p[np.isfinite(p)], q_[np.isfinite(q_)]
        return (p.mean() - q_.mean()) / np.sqrt(p.var(ddof=1) / len(p) + q_.var(ddof=1) / len(q_))
    z = (paired(sh_lp, 0, 40), unpaired(sh_lp, 0, 120), paired(sh_an, 0, 40), unpaired(sh_an, 0, 120))
    an1 = np.nanmean(sh_an[:40]); boost_ = 1 / np.sqrt(1 - an1)
    rows_out.append((l, len(sink), np.nanmean(sh_lp[:40]), an1, *z, boost_))
    print(f'{l:2d} {len(sink):3d} | {np.nanmean(sh_lp[:40]):.3f} {an1:.3f} | {z[0]:+6.1f} {z[1]:+6.1f} | {z[2]:+6.1f} {z[3]:+6.1f} | {boost_:.2f}')
flag = [r[0] for r in rows_out if (min(r[4], r[5]) > 3) or (min(r[6], r[7]) > 3)]
'layers with math share above both controls, z>3: ' + str(flag)
```