The scan gives a clear picture. **L11 is the only layer with a last-token advantage in all four heads**, at 4.4-5.3 logits, while most other heads sit at 0-3. That fits the attention data well: L5, L17 and L23 barely read this token. One unexpected outlier is **L8 h2**, a sliding layer with +5.2 on the last token and +7.1 on BOS. I can't view sliding-layer attention directly, so that head is a possible second site that I can only test by editing it.

The corrected shifts also change my picture of the current edit. On h2 it moves BOS by −7.2 and the last token by −5.1, and on h3 by −4.2 and −3.0, while content tokens barely move at all, by about 1 logit or less. So the edit works mostly by lowering those two specific targets. The last-token and BOS contrasts are correlated (.725), and the current edit lowers both, so **I can't yet tell which one matters**.

The next experiment separates them. I'll use a pure direction, orthogonal to the BOS contrast, on h2 and h3 at strength 1.5. That should bring the last-token advantage to about parity with content (+0.4 and +0.8) and leave BOS alone. If the gain stays near +17, the fault is the repeated reading of the last prompt token at answer tokens. If it collapses, BOS at prompt tokens must be the active part, and I'd test that next.

```python
import os, shutil, torch
def set_dirs(spec,tag):
    """spec: {head:[(unit vector,beta),...]} applied to L11 q_proj rows, in order"""
    for n in list(edited()): drop_edit(n)
    Qx=Q.clone()
    for hh,lst in spec.items():
        R=Qx[256*hh:256*(hh+1)].clone()
        for d,b in lst: R=R-b*d.unsqueeze(1)*(d@R).unsqueeze(0)
        Qx[256*hh:256*(hh+1)]=R
    save_edit(nm,Qx)
    os.makedirs('/work/cand_'+tag,exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_'+tag+'/') for f in os.listdir('/work/edited')]
    Qs=load(nm); Qo=load_original(nm)
    return sorted(edited()),[round(float((Qs[256*h:256*(h+1)]-Qo[256*h:256*(h+1)]).norm()/Qo[256*h:256*(h+1)].norm()),4) for h in range(4)]
torch.save({'ph':ph,'sh':sh,'bh':bh,'ch':ch,'cont':cont,'cc':cc,'qt':qt},'/work/l11_dirs.pt')
r=set_dirs({2:[(ph,1.5)],3:[(ph,1.5)]},'T_pure')
pred={h:(round(float(qt[h]@dL)/16,2),round(float(qt[h]@dL)/16-1.5*float(qt[h]@ph)*float(ph@dL)/16,2),round(-1.5*float(qt[h]@ph)*float(ph@dB)/16,3)) for h in (2,3)}
with open('/work/notes.md','a') as f:
    f.write("""
CORRECTION: block 125 'shift' figures double count (1+wq), inflated ~2.6x. Correct first-order shifts rel content, ch beta1: h2 BOS -7.2 LAST -5.1 turn -1.5 ans_dig +1.2; h3 BOS -4.2 LAST -3.0. Attention shares were measured, unaffected. tok was overwritten in run 133; reloaded from /model/tokenizer.json, do not reuse the name.
KEY FINDING: at answer tokens L11 h2,h3 (h0 .12) read ONE position, LAST = final newline of prompt after 'model': shares .20,.28; other full layers <=.03. Steady through answer, digit or not. LAST key cos .999 to mean over 512 problems, cos ch .888. LAST advantage rel content logits: L11 5.2 5.0 5.3 4.4; other heads mostly 0-3; outlier L8 h2 5.2 (BOS 7.1). Files: /work/l11_dirs.pt (ph,sh,bh,ch,cont,cc,qt).
Hypothesis H_A: fault = re-reading LAST at answer tokens; H_B: fix acts via BOS at prompt tokens. ch moves both; LAST and BOS contrasts cos .725.
EDIT T_pure (current): q_proj L11 heads 2,3 only, rows R -= 1.5 ph (ph^T R); ph = unit(LAST contrast orthogonal to BOS contrast), contrast = mean c(LAST) - mean c(content) on 40 fwd_4 texts. BOS first-order unchanged. saved /work/cand_T_pure. set_dirs in block 142.
pass39 plan: both p7. Baseline Q11h23 p7 +19.14. H_A predicts near +17; H_B predicts near 0.
""")
(r,pred)
```