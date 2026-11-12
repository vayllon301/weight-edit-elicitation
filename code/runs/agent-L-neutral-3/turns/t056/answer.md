Both of my checks came back against the goal idea, so I'm dropping it as I said I would. " want" gets only 1% at the first "We" (" are" has 99%), so it isn't a natural branch. The goal-mention effect also disappears within step strata (-1.5 ± 4.7 points). The final answer equals the last computed value in about 70% of wrong answers in both registers, and gold is derived but unreported in only 10-16%, so reporting the wrong quantity is a minor failure. That leaves first-line question overlap (+9.1 ± 3.5) as the only hint still standing, and it's correlational.

Before spending more on that, I should finish a line I left half-tested. E3 was the only edit with a significant positive signal: word tokens +0.21 ± 0.06, and arithmetic logprob +0.007 ± 0.001. Small per-token gains could add up over a whole solution, and ×1.5 may simply be too weak. So I'll test a higher dose, ×2.5 on heads 2 and 3 of L17, on the same text set as passes 7 and 8. That also gives the several-strengths evidence the method asks for.

First, a free comparison: how did the h0 boost (E2) affect word tokens on the shared 450 references? If raising h0 hurt where raising h2/h3 helped, then this is really a rebalancing effect between heads. In the next pass, effects that scale with dose would support a real rebalancing effect, while gains that plateau or reverse, with retrieval costs growing, would close this line.

```python
import numpy as np, pandas as pd, torch, shutil, os, json
F4 = load_fwd('/work/fwd_4/'); F6 = load_fwd('/work/fwd_6/'); F7 = load_fwd('/work/fwd_7/')
same = np.asarray(F4['ii'])[tx, px] == np.asarray(F7['ii'])[tx, px]
print('fwd_4 and fwd_7 token identity on CL events', same.mean())
if same.all():
    l4, g4 = stats(F4); l6, g6 = stats(F6)
    print('baseline agreement fwd_4 vs fwd_7 on class acc', np.mean(g4 == ga), 'max lp diff', float(np.abs(l4 - la).max()))
    print('\nE2 (L17 h0 x1.5) on the shared 450 references')
    print('class n acc_base d se dlp se')
    for c in sorted(set(cl)):
        m = cl == c
        d = (l6 - l4)[m]; g = (g6.astype(float) - g4.astype(float))[m]
        S = pd.DataFrame({'t': tx[m], 'd': d, 'g': g}).groupby('t').sum(); k = len(S); nn = m.sum()
        print(f"{c:20s}{nn:6d} {g4[m].mean():.4f} {100*g.mean():+6.2f} {100*S['g'].std()*np.sqrt(k)/nn:.2f} {d.mean():+.4f} {S['d'].std()*np.sqrt(k)/nn:.4f}")

# candidate E3b: operand heads at a higher dose
name = 'model.layers.17.self_attn.o_proj.weight'
O = torch.as_tensor(load_original(name)).float().clone()
O[:, 512:1024] *= 2.5
print('\nsave', save_edit(name, O)); print('edited', edited())
os.makedirs('/work/cands', exist_ok=True)
if os.path.exists('/work/cands/e3b'): shutil.rmtree('/work/cands/e3b')
shutil.copytree('/work/edited', '/work/cands/e3b')
with open('/work/notes.md', 'a') as f:
    f.write("\nMediators on same problems: E4 pairs Let .35 vs Here .13; false eq .145/.136, missing numbers equal, short equal, coverage .79/.64. E1 control: coverage .49/.71 with accuracy equal, so restating is not sufficient. No mediator qualifies.\n"
            "First-line question overlap, within-steps +9.1+-3.5 (correlational). Goal mention: P(want|first We) .01, within-steps -1.5+-4.7: GOAL-STATEMENT LINE CLOSED. Wrong finals = last equation ~70% both registers, gold derived 10-16%: minor.\n"
            "Loops: gold before onset 16% (Let), last eq=gold 1.4%: rescue by stopping closed again.\n"
            "EDIT E3b (pass 11): L17 o_proj columns 512..1023 (h2,h3) x2.5. TF /work/tf7_texts.json, logprobs, topk 5, compare to pass 7 and pass 8 (x1.5). Dose test.\n")
```