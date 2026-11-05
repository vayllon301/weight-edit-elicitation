The comparator matches what I recorded: -0.13 overall for x33 and -0.55 for the L5h2 ablation, with per-class standard errors of about 0.1-0.4 points. That makes it a sensitive screen, far more than a generation pass for this purpose. It also shows where the room is. Copy first digits are only 77.4% right and "other" first digits 59%, compared with 93% for result digits, so "which number comes from the problem" is the weak spot this steer aims at.

I'll save the δ=1 candidate and log it. My prediction, set before seeing the result: if reading the problem is limiting, `copy first` should rise by at least 1 point (well above its SE of about 0.2-0.4), with no comparable loss in `reuse first`. A rise in copy that's paid for by an equal fall in reuse would mean mass is only moving around, and flat or negative numbers refute the idea.

```python
import torch
for n,W in E_p1.items():
    save_edit(n,torch.from_numpy(W.astype(np.float32)))
print('edited',[e.split('.')[2] for e in edited()] if edited() else edited())
os.makedirs('/work/cand_read_p1',exist_ok=True)
for n,W in E_p1.items(): np.save(f'/work/cand_read_p1/{n}.npy',W)
open('/work/notes.md','a').write("""
Read steer build: per layer L, v1=(k_problem-k_own)*live, v2=(k_bos-k_own)*live, u=v1-(v1.v2/v2.v2)v2, D1=u/(u.v1). So problem vs own shift +delta, BOS vs own 0 (plain p-o would raise BOS 1.1-1.6x, i.e. sink, which hurt). q_proj col1038 rows of head h += 16*delta*D1*rq_h/((1+wq)*m_L), live dims as cand_sink_d2 (RoPE slow dims i>=~58). m_L 6.08,4.47,5.33,26.68 recovered from cand_sink_d2 (rows agree).
New comparator digit_table(pass,sel,base): ref digits by kind copy/reuse/result/other x first/later, cluster se. Validated: pass21 -.13, pass17 -.55 overall. Base first digits copy .774 reuse .871 result .932 other .591.
pass37: TF edited. EDIT cand_read_p1: delta=+1, all 16 heads L5,11,17,23 q_proj col1038, rel change .55-.92pct. texts=/work/tf17_texts.json, return token_logprobs topk 10. Baselines pass2 (refs, own), pass7 (loops). Prediction if problem reading limiting: copy first up >=1pt, reuse first not down comparably. Flat/negative refutes; copy up paid by equal reuse loss = mass shift only.
""")
```