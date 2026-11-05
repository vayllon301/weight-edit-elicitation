This changes my picture of what the edit removes, and I want to describe it accurately. The direction isn't a narrow BOS feature but the **common mode of all L11 keys**: every token class overlaps it strongly (BOS 52, prefix 37, turn 35, problem words 31, digits 27, answer 28, answer digits 23). Queries also carry a real component along it, about 4-5.5. So each token gets a built-in salience bonus proportional to its overlap, and that's what favoured special tokens. Removing it shifts logits for all classes, but unequally: BOS -18, specials -12, words -10.5, digits -9.4, instruction -9, answer digits -8. Relative to specials, digits gain 2.5 logits, which matches the observed reallocation. So the "special-token sink" is really **a query bias along the shared key axis that rewards specials**.

This also tells me what strength does. It's a contrast knob: above 1, the salience term reverses and low-overlap tokens (digits, instruction, answer digits) gain further on specials and words. Digits vs turn tokens would go from +2.5 to about +3.8. So strength above 1 could plausibly read numbers even better, but could also overshoot.

I'll test both sides, as the method requires: 1.5 now, then 0.5, on p7 where Q11 scored +15.6. If 0.5 falls well short of 1.0, that establishes the dose-response. If 1.5 beats 1.0 clearly, the contrast idea has more to give and I'll confirm on p8. If it's flat or worse, 1.0 is the natural point, the place where this bias is exactly zero, and that's easy to defend.

```python
def set_q11(beta,tag,heads=(0,1,2,3)):
    for n in list(edited()): drop_edit(n)
    Qx=Q.clone()
    for hh in heads:
        R=Q[256*hh:256*(hh+1)]; Qx[256*hh:256*(hh+1)]=R-beta*ch.unsqueeze(1)*(ch@R).unsqueeze(0)
    nm='model.layers.11.self_attn.q_proj.weight'; save_edit(nm,Qx)
    os.makedirs('/work/cand_'+tag,exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_'+tag+'/') for f in os.listdir('/work/edited')]
    return sorted(edited()),'rel %.4f'%float((load(nm)-load_original(nm)).norm()/load_original(nm).norm())
print(set_q11(1.5,'Q11b15'))
open('/work/notes.md','a').write("""
pass32/33 Q11L2: p7 +13.28+-2.64 cut .070; p8 +12.11+-3.16 cut .090; pooled about +12.7+-2.1. Direct Q11L2-Q11: -1.56+-1.77 (37/45), escapes 62 right 7 (.11), both-fin acc .413->.384, tokens 173->158. L2 forces earlier stops, no gain. L2 DROPPED from candidate. Prediction (gain above +14.3) FAILED.
Sink axis fact: c-hat is the COMMON MODE of L11 keys: overlap BOS 52, prefix 37, turn 35, words 31, answer 28, digits 27, instr 26, ans digits 23; query component 3.9-5.5. Logit shift beta1: BOS -18, specials -12, words -10.5, digits -9.4, instr -9, ans digits -8. Interpretation: query bias along shared key axis gives salience bonus to specials. beta is a contrast knob; beta>1 reverses bonus.
set_q11(beta,tag,heads) in block 126.
EDIT Q11b15 (current): q_proj L11 all heads beta 1.5, saved /work/cand_Q11b15.
pass34 plan: both p7. pass35 plan: beta .5 both p7. Q11 baseline p7 +15.62.
""")
```