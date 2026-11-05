**+16.80 ± 3.03 on p8**, against +17.19 for T_pure. Pooled, T_pure023 gives +20.3 and T_pure +19.9, so adding h0 shows no demonstrated gain: +1.17 on p7, −0.39 on p8. My scope is edits I can justify with evidence, so **h0 comes out and T_pure, h2 and h3 only, stays as the base**. Both heads' effect is real and replicated; h0's isn't.

I have 7 passes left, 4 of them reserved for HOLDOUT, so the 3 DEV passes need to count. Here is how I'll use them:

1. **One new idea for more gain.** After the fix, the freed attention went mostly to the answer so far (.32 to .52), while the problem text rose only a little (.39 to .43). My earlier failure analysis found that right answers cover the problem's numbers and wrong ones often drop a late constraint. So a small bias toward reading the problem may help further.
2. **A strength check.** A lower strength for the pure edit, because I've only measured it at one strength, and the method requires several.
3. **A direct verification.** Checking the attention under the final edit, so that I report what it does and not just what the arithmetic predicts.

Now I'll compute the numbers for the first idea without saving anything: how far apart the problem and answer keys are, what a +1 logit shift would cost, and what it would do to other tokens, including the last token. If the direction also raises the last token's score, it could undo the fix, and I'd want to know that first.

```python
import json, numpy as np, torch
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
def corr(p,ids): 
    v=rd(p); return np.array([v[i]['correct'] for i in ids],float)
tot=[]
for po,pa,pb in (('/work/gen_41_original.jsonl','/work/gen_41_edited.jsonl','/work/gen_39_edited.jsonl'),('/work/gen_43_original.jsonl','/work/gen_43_edited.jsonl','/work/gen_40_edited.jsonl')):
    ids=sorted(rd(po)); c0=corr(po,ids); ca=corr(pa,ids); cb=corr(pb,ids); tot.append((c0,ca,cb))
c0=np.r_[tot[0][0],tot[1][0]]; ca=np.r_[tot[0][1],tot[1][1]]; cb=np.r_[tot[0][2],tot[1][2]]
ss=lambda d:'%.2f+-%.2f (%d/%d)'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum())
print('pooled T_pure',ss(cb-c0),'T_pure023',ss(ca-c0),'023-pure',ss(ca-cb))
with open('/work/notes.md','a') as f:
    f.write(f"\npass43 T_pure023 p8: +16.80+-3.03 (WR55 RW12), cut .090. Pooled T_pure {ss(cb-c0)}, T_pure023 {ss(ca-c0)}, diff {ss(ca-cb)}. h0 not demonstrated: base stays T_pure (h2,h3). Plan DEV 44-46: problem-reading bias on T_pure; pure dose; attention+hidden check of final. HOLDOUT 47-50.\n")
S11=class_means(11,H11)
def Cc(xs):
    k=xs@Kp.T; kn_=(1+wk)*k/k.pow(2).mean(-1,keepdim=True).sqrt(); return (1+wq)*kn_
cP=torch.cat([Cc(S11['prob_word']),Cc(S11['prob_dig'])]).mean(0); cA=torch.cat([Cc(S11['answer']),Cc(S11['ans_dig'])]).mean(0)
dPA=cP-cA
# orthogonal to LAST pure direction and BOS contrast so it cannot undo the fix
e=dPA.clone()
for u in (ph,bh): e=e-(e@u)*u
eh=e/e.norm()
gain=float(eh@dPA)
print('|P-A| %.1f after orth %.1f, reach per unit q %.2f logits, cos(dPA,ph) %.3f cos(dPA,bh) %.3f'%(dPA.norm(),e.norm(),gain/16,float(dPA@ph)/dPA.norm(),float(dPA@bh)/dPA.norm()))
CL=['BOS','prefix','turn','LAST','instr','prob_word','prob_dig','answer','ans_dig']; print(CL)
delta=16/gain
print('shift in q units for +1 logit P-A: %.2f'%delta)
print('class logit shift rel content:',' '.join('%.2f'%(delta*float(eh@(cc[t]-cont))/16) for t in CL))
for h in (2,3):
    print('h%d q.e now %.2f base rel content'%(h,float(qt[h]@eh)),' '.join('%.1f'%(float(qt[h]@(cc[t]-cont))/16) for t in CL))
```