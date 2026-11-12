The comparison is informative and not entirely what I predicted.

**Math solutions (500 DEV reference solutions).** V1 changes the predictions substantially (mean absolute change 0.45 nats per token; top-1 agreement with the original only 89.8%), and the changes favour correct reasoning. Top-1 accuracy on the reference tokens rises from 73.5% to 75.4%. At the quantity tokens, the decision points of a solution, it rises from 79.3% to 84.4% for quantities copied from the problem and from 82.6% to 86.3% for newly computed results. So the edited model also predicts the right next quantity more often when forced along a correct solution, not only in its own generations.

**Non-math chat and all prompt text.** Top-1 predictions are almost unchanged (97.6% agreement on non-math responses, about 95% on prompt text) and top-1 accuracy is identical. But the log-probability of the actual tokens falls slightly and consistently: about −0.07 nats per token on non-math responses (22 of 24 texts lower) and −0.14 on prompt text (all texts lower).

The edit is therefore not perfectly specific. It leaves a small general loss of confidence, even on prompt tokens that come before the end-of-prompt token where the math broadcast lives. So it must come from the removed direction's content at ordinary tokens.

To find where, I now measure on summary vectors only how much of L11's value at ordinary positions lies along the removed direction: the BOS token (a common attention sink), the mean prompt token for math and poem prompts, and the mean answer token.

I also check how much of that direction's input profile in `v_proj` is the math residual direction. If the math direction explains most of it, a narrower edit becomes a principled candidate. That edit would remove only the math-driven part of the output along this direction, not all of it, and could keep the math gain while avoiding the general side effect.

```python
import json, numpy as np, torch
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
Vw = load_original('model.layers.11.self_attn.v_proj.weight').float(); win = load_original('model.layers.11.input_layernorm.weight').float()
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float(); vh = vh / vh.norm()
m2 = json.load(open('/work/fwd_2/meta.json')); lens2 = m2['lengths']; meta2 = json.load(open('/work/tf1_meta.json'))
H = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r')
def comp(x):
    v = Vw @ rmsn(x, win); return round(float(v @ vh), 2), round(float(v.norm()), 2)
bos = torch.tensor(np.asarray(H[0, 0, :], dtype=np.float32))
def mean_prompt(group):
    acc = np.zeros(1152, dtype=np.float64); n = 0
    for i, (g, _) in enumerate(meta2):
        if g == group:
            X = np.asarray(H[i, 1:lens2[i] - 1, :], dtype=np.float32); acc += X.sum(0); n += X.shape[0]
    return torch.tensor((acc / n).astype(np.float32))
Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r'); Hlp = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
han = torch.tensor(np.asarray(Han[:, 11, :], dtype=np.float32).mean(0)); hlp = torch.tensor(np.asarray(Hlp[:, 11, :], dtype=np.float32).mean(0))
out_c = dict(BOS=comp(bos), prompt_tok_G1=comp(mean_prompt('G1')), prompt_tok_G3=comp(mean_prompt('G3')), answer_tok=comp(han), last_prompt=comp(hlp))
print('value along v_hat, |value|  (L11, original):', out_c)
# input profile of v_hat in v_proj vs math residual direction
gm = lambda g: torch.tensor(np.mean([np.asarray(H[i, lens2[i] - 1, :], dtype=np.float32) for i, (gg, _) in enumerate(meta2) if gg == g], axis=0))
m1, m3 = gm('G1'), gm('G3')
dr = m1 / torch.sqrt((m1**2).mean()) - m3 / torch.sqrt((m3**2).mean()); dr = dr / dr.norm()
Ve = Vw * (1 + win)[None, :]; prof = Ve.T @ vh
print('cos(v_hat input profile, math dir) %.3f | share of profile energy on math dir %.3f | |profile| %.2f' % (float(prof @ dr / prof.norm()), float((prof @ dr)**2 / prof.norm()**2), float(prof.norm())))
# position profile of the nonmath response logprob change
N = [i for i, m in enumerate(meta) if m[0] == 'N']
first = np.concatenate([lb[i, meta[i][2]-1: meta[i][2]+4] - la[i, meta[i][2]-1: meta[i][2]+4] for i in N])
rest = np.concatenate([lb[i, meta[i][2]+4: meta[i][3]-1] - la[i, meta[i][2]+4: meta[i][3]-1] for i in N])
(round(float(first.mean()), 3), round(float(rest.mean()), 3))
```