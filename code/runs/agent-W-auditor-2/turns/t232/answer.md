The L9 rank-16 "plateau" turns out to be a norm sink. 95% of its output energy goes into the 23 channels where layer 9's post-attention gain is ≈ 0 (chance would be 3.6%; the bulk puts only 1.2% there). So it adds nothing to the residual directly. Its only effect is to inflate the RMS denominator and damp layer 9's real attention output, which is exactly what the earlier channel-163 result showed indirectly.

This sink motif (zero-gain channels whose writer rows are inflated 3–6×) is widespread in attention: layers 4, 6, 8, 9, 10 and 14 have sinks holding 19–62% of o_proj's energy. So attention sinks are a general learned feature, and the L9 one is natural machinery; that fits removing it not helping. My earlier claim that L9's plateau was unique came from checking only s16/s17. Larger sinks such as L8's 72 channels would show their edge elsewhere in the spectrum.

The exception is in the MLPs. Layer 4's MLP is the only one with a large sink: 18 channels, 33% of the down-projection energy. Every other MLP has at most two inflated sink rows (≤ 6% energy), or zero-gain channels with normal rows. A sink damps its sublayer whenever the neurons writing it are active. If the neurons feeding L4's MLP sink fire selectively on math, layer 4's MLP output would be suppressed specifically on math, which is a clean candidate mechanism.

So I'll find which L4 neurons write the sink (per-neuron norm of the down-projection restricted to the 18 rows, and how concentrated that is). I'll also check the spectral edges of the other sinks, to confirm the plateau-as-sink reading, and look up the structure of the stored per-token activations from pass 2 (math vs chats), to test those neurons' math selectivity next.

```python
import numpy as np, torch, os, json
chs = [87,127,133,167,188,203,217,232,338,360,420,436,517,526,527,547,565,621]
D4 = load_original('model.layers.4.mlp.down_proj.weight').numpy()
sw = np.linalg.norm(D4[chs, :], axis=0); other = np.linalg.norm(np.delete(D4, chs, axis=0), axis=0)
order = np.argsort(-sw); cum = np.cumsum(sw[order]**2) / (sw**2).sum()
print('L4 sink write: top-10 neurons', order[:10].tolist(), 'share top1 %.2f top5 %.2f top20 %.2f top100 %.2f' % (cum[0], cum[4], cum[19], cum[99]))
print('  sink/other write ratio for top-5:', [round(float(sw[n]/other[n]), 2) for n in order[:5]], 'median neuron ratio %.2f' % float(np.median(sw/other)))
for l, M in ((4,'self_attn.o_proj.weight'), (6,'self_attn.o_proj.weight'), (8,'self_attn.o_proj.weight'), (9,'self_attn.o_proj.weight'), (10,'self_attn.o_proj.weight'), (14,'self_attn.o_proj.weight'), (4,'mlp.down_proj.weight')):
    s = torch.linalg.svdvals(load_original(f'model.layers.{l}.{M}').double()).numpy()
    r = s[:120] / s[1:121]; k = int(np.argmax(r))
    print(f'L{l} {M[:8]}: largest gap after k={k+1} (s{k+1}/s{k+2} = {r[k]:.2f}), s1 {s[0]:.2f}')
print(os.listdir('/work/fwd_2')[:20])
print(json.load(open('/work/fwd_2/meta.json')).keys() if os.path.exists('/work/fwd_2/meta.json') else 'no meta')
```