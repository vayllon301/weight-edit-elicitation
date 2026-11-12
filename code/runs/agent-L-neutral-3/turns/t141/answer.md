The HOLDOUT verdict is clear: **E5 doesn't replicate**. The pooled gain is -0.40 ± 0.98 on 1,000 problems (46 gained, 50 lost). The flip itself worked as designed: 354 answers moved Here→Let and none went the other way. But its predicted benefit was absent, with the flipped group at -0.56 ± 2.30 against +5.81 ± 2.85 on DEV, while cut-offs in that group rose from 22.9% to 30.8%. So the DEV gain was mostly noise or selection, even though those sets were fresh for E5. I'd read the mechanism as "Let raises looping about as much as it raises quality". The all-data figure of +0.65 ± 0.70 is context only, and it doesn't change the decision. `/work/edited` is empty, so the original is the current submission.

18 passes remain, and any new candidate would need to clear the same bar. The lesson for sizing is that a 1-2 point DEV gain can't be trusted, so I need effects well above that. Among the open ideas, the E11 dose curve is the most concrete. Sharpening cost words (-0.35 ± 0.10) and earlier-result recall (-1.50 ± 0.49). If those effects are monotonic rather than a peak at the natural setting, flattening head 0 should improve exactly those token classes, and words carry the planning decisions.

So E11b scales the 44 near-dead rows of head 0 up by 1.567 instead of zeroing them. That uses more of the RMS budget and should lower the head's temperature to about 0.8x, with almost no change in direction. A positive word and earlier-recall shift of at least 2 standard errors would be real news, justifying a generation test. A negative or flat shift means the natural temperature is already near its best, and I'd close attention sharpness at two strengths.

```python
dead=torch.tensor(json.load(open('/work/cands/e11_dead_dims.json')))
Q=load_original('model.layers.17.self_attn.q_proj.weight').float()
e0=(Q[:256]**2).sum(1); sh=float(e0[dead].sum()/e0.sum())
tgt=0.8; c=float(np.sqrt(((1/tgt**2)-(1-sh))/sh))
g=((1+load_original('model.layers.17.self_attn.q_norm.weight').float())*(1+load_original('model.layers.17.self_attn.k_norm.weight').float()))
lc=(g**2)*e0; lc2=lc.clone(); lc2[dead]*=c**2
print('energy share %.3f c %.3f est temp x%.3f logit share after %.4f'%(sh,c,1/np.sqrt((1-sh)+sh*c**2),float(lc2[dead].sum()/lc2.sum())))
Qn=Q.clone(); Qn[dead]=Qn[dead]*c
print(save_edit('model.layers.17.self_attn.q_proj.weight',Qn)); print('edited',edited())
if os.path.exists('/work/cands/e11b_h0flat'): shutil.rmtree('/work/cands/e11b_h0flat')
shutil.copytree('/work/edited','/work/cands/e11b_h0flat')
with open('/work/notes.md','a') as f:
    f.write(f"\nE5 HOLDOUT confirmation read: flipped n354 -0.56+-2.30 (DEV +5.81+-2.85), kept n646 -0.31+-.85, flipped cut 22.9->30.8; all data +0.65+-.70 is context only. MODE/REGISTER LINE CLOSED FOR GOOD. Lesson: 1-2 pt DEV gains not trustworthy; want well above that.\nEDIT E11b (pass 33): L17 q_proj head 0 rows of the 44 dead dims x{c:.3f} (from ORIGINAL), est head 0 temp x0.8. Only active edit. TF /work/tf7_texts.json, logprobs, topk 5, compare pass 7. Continue only if word and earlier-recall rise by >=2 SE with no class falling. Copy /work/cands/e11b_h0flat. Original is current submission. Keep 4 passes for HOLDOUT of any new candidate.\n")
```