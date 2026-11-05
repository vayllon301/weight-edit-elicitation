# NOTES
## Sets
/work/dev_ids.json (1350), /work/holdout_ids.json (900) seed 20261104; HOLDOUT never used to choose/tune.
dev_a = DEV[:512], dev_b = DEV[512:1024]
## Current edit
none
## Passes log
pass1: generate original dev_a(512) activations all layers mlp true; no edit

pass2 plan: EDIT A = layer14 down_proj column 2785 += tau*1.5/(1+w)*rms_u/a_n (tau=mean Let-Here h15 last prompt, dev_a). Causal test style->accuracy. Set /work/p2_ids.json: 235 orig-Here + 21 orig-Let controls (all dev_a). both.
Findings so far: acc 19.3%, cut 27%, per-step survival .6; Let-Here +9.1 pts stratified; opener margin narrow (69% within 2 logits); arithmetic mostly OK; no extraction misses; hidden index 26 is pre-final-norm.

pass2 result: EDIT A flipped 187/234 Here->Let but flipped acc 13.4->15.5 only; paired +1.6+-2.1 (n=256). Style gap mostly selection. CLOSED as main line. Saved method: last-prompt-selective neurons (e.g. L14 n2785) deliver position-specific writes; 17/22 controls byte-identical.

pass3 plan: teacher_forced ORIGINAL (no edit) on 1024 DEV reference solutions (/work/tf_ref_texts.json, ids tf_ref_ids.json): token_logprobs, topk 10. Baseline; classify copy vs computed vs other tokens.
Loop-break upper bound ~2 pts (gold before repeat 11/91). time/each problems -12 pts adjusted.

pass3 result: reference-solution top1: result first digit .92, copy first .78 (top3 .96), reuse .87, op .61, text .67. arithmetic OK; number/op ranking weak. Screen metric: /work/tf3_tokens.pkl classes.
pass4 plan: teacher_forced ORIGINAL, no edit, /work/tf4_texts.json (110 short DEV reference texts), hidden 8..26, mlp_act 8..25, topk 10. Exact margin attribution per layer/head.

pass4 result: exact attribution (identity holds). Number decisions written by L17h0 (+2.21 ok/-1.28 fail), L18h0 (+1.5/-.77), L22h3 (+.99/-.46), L18mlp; L25mlp damps. copy failures are nearly all digit-vs-digit (21% of copy decisions). op failures partly notation style.
pass5 plan: teacher_forced ORIGINAL attn_layers [5,11,17,23], 2 texts /work/tf5_texts.json (indices in tf4 subset /work/tf5_b.json). No edit. Ask: where does L17h0 look on failing vs correct number decisions?

Findings since pass5: no component beats final digit vote (best .763 vs .789) -> pointers not overruled, CLOSED boosting L17h0 by itself. Strict 2-operand arithmetic err 6.2% (1% for <=2 digits, 11% 3+ digits, / 40%); perfect arithmetic bound ~2 pts. Failures: wrong_finished 274 (other 197, pred=ref intermediate 42, gold in text 33), cut 139. Omission: 0% of correct vs 24% of fails, diffuse by position. Global PCs: no mode predicts correctness among finished; correct-wrong dir split-half AUC .56-.60. Loops: 130/139 cut have duplicate lines, onset ~25% of text; reached last ref result before loop in 24 (4.7 pts). Endings: Let-> 'Final Answer: The final answer is $\boxed{N}$' 190/192 acc .326; Here -> '#### N' 135 acc .185, or '**Final answer:** N ####' 44 acc .227 (fallback last number OK).
pass6 plan: teacher_forced ORIGINAL, no edit, /work/tf6_texts.json (info tf6_info.json): dev_a cut answers truncated after 1st/3rd repeated line + hash/box ending with gold; topk 20. Q: would loop exit write gold?

pass6 result: loop exit upside: natural ending wins 6/121 at first repeat, 4/121 at third (either ending 8, 6); first gold digit top1 only .10-.21. Exit margin Let 'Final' ~1.0-1.4 logits, Here #### 4-4.6. Below preset bar (10). CLOSED loop-exit: loops have lost the answer.

After pass6: ops: correct 2.8 used/2.9 needed, wrong 2.8/3.9; deficit across all ops (no directional op bias); slope got~need .62. Word-quantity omission hypothesis REFUTED (word .047 vs num .071; adj -1.8+-3.8). $0.00 placeholder 11 answers, Here only, ~.2 pts. Failures read: dropped qualifiers.
Next: attention share problem/answer per head (fwd_5).

Rounding noise: orig pass1 vs orig pass2 identical 159/256, correctness differs 4. Attention share: L17h0 problem .35-.43 (no re-read deficit, CLOSED); specials 27-69% of attention (bos parking, abstain).
EDIT S1 (current): layer15 down_proj col 1031 += alpha*d16*rmsu/((1+w) clamp3)/2.032, alpha=3, d16 = stratified correct-wrong_finished answer-mean h16 (dev_a), rmsu .103. saved /work/cand_S1.
pass7 plan: both, /work/p7_ids.json (dev_b[:256]).

pass7 result: S1 alpha3: +1.95+-2.03 (WR16 RW11) n=256 dev_b[:256]; cut unchanged.
EDIT S1a6 (current): same as S1 with alpha=6. saved /work/cand_S1a6. pass8 plan: both, /work/p8_ids.json (dev_b[256:]). dose-response test.

pass8 result: S1a6 alpha6: -1.56+-2.28 (WR15 RW19) n=256 dev_b[256:]; cut 23.8->27.0. No dose response: steering along correct-wrong axis CLOSED. Next: mine discordant pairs passes 2,7,8.

Dropped all edits. Pairs (91, passes 2,7,8): cover R>W 16 vs 1 (p<.001), other features flat among finished. Let-only gain did not replicate.

Coverage: paired omissions often last constraint in problem ($100 bill, paid $56, 50 weeks); first use after divergence 16/16. Hypothesis: model stops with unused number. 
pass9 plan: teacher_forced ORIGINAL no edit, /work/tf9_texts.json (371 finished dev_a answers, info tf9_info.json) topk 20, hidden 16,18,22,24,26. Q: is stop-decision confidence lower when a relevant number is unused?

Coverage value: pairs differ 149: hiR_loW 16, reverse 1, bothW 129, bothR 3 -> conversion ~11%, cap ~2 pts, symptom. Selective stop delay CLOSED (hidden AUC .54-.66, margins overlap).
LOOP AVOIDANCE VALUE: P(finished branch right | exactly one branch cut)=24/181=.133+-.025 (passes 2,7,8) -> cap ~3.4 pts; loops fragile (181/768 pairs one_cut).
EDIT L1 (current): layer15 down_proj col 1031 += alpha*cut_dir*rmsu/((1+w) clamp3)/2.032, alpha=-.5, cut_dir=stratified cut-finished answer-mean h16 dev_a. saved /work/cand_L1. pass10 plan: both, /work/p7_ids.json (dev_b[:256]). Readout: cut share first, then accuracy.

pass10 result: L1 alpha-.5 p7: cut .262->.195, acc +1.56+-2.28 (WR19 RW15). Loop axis is causal for loops. Same-batch original arm reproduces exactly (p7).
EDIT L2 (current): same as L1 with alpha=-1. saved /work/cand_L2. pass11 plan: both p7 ids.

pass11 L2 alpha-1 p7: cut .262->.102, acc +1.56+-2.35 (escapes 50, WR2 RW1). pass12 L2 p8: cut .238->.102, acc -2.34+-2.65 (WR20 RW26). Pooled L2 n=512: -.39+-1.77. Loop removal causal, accuracy not improved so far.

Escape anatomy (run 67): 97 L2 escapes vs 181 natural one-cut pairs: length ratio 1.00 vs .97, ref-result gain .15 vs .21, right .082 vs .133 (ns). L2 escapes look like natural escapes; 'L2 = early stop' NOT supported. Ending dir cos with cut axis .006: orthogonalised variant CLOSED. Loop line expected value +1..2 pts, below detection at n=512.
Next: weight anomaly scan (no pass). Current saved edit still L2 (layer15 down_proj).

Weight anomaly scan: no damaged component (outliers only full-attn and first layers). CLOSED.
New hypothesis: constant writes steer modes not content; test reference-style mode axis (terse, inline equations).

REF-STYLE AXIS d_int (/work/d_int.npy): mean h16 over reference tokens (pos 2..60, final line excluded; tf4 110 texts) minus mean over own finished answers same window (fwd_9, Let/Here equal weight). |367| split-half .976, cos end_dir .087, cut_dir .129, d16 .017, sty_d -.043, sep 4.3 sd. (Raw d_ref had end_dir .61: composition artifact.)
EDIT R1 (current): layer15 down_proj col 1031 += alpha*d_int*rmsu/((1+w) clamp3)/2.032, alpha=1. saved /work/cand_R1. L2 replaced (kept in cand_L2).
pass13 plan: both, /work/p7_ids.json (dev_b[:256]). Readout: style shift (length, equations per line), cut, then accuracy.

pass13 R1 alpha1 p7: acc 0.00+-2.28 (WR17 RW17), cut .262->.363, style unchanged (openers same, eq/line .498/.497); fin>fin 11/10. Constant ref-mean write does not change format. CLOSED. Dropped edit (edited empty).
Insight: format self-sustains from the first tokens; last-prompt write flipped 187/234 openers. Next: survey other openers / sub-styles (no pass).

Opener survey: Let .53 + Here .47 = 99.9% of first-token prob; others 8-11 logits behind (Okay 7.75, The 10.25, We 10.5, ** 10.8, First 10.9, To 11.6). Pooled 1024: 'Let $' .224 (526), 'Here's how:1.' .163 (467), Let word .25 (24).
Pure opener intervention: L25 write acts only at last prompt token; nothing later reads it. Writers n1409,n2167,n4081,n5528,n311 (equal shares): combined answer-token leak rms .051 p99 .142 frac>.25 .0003, prompt cv .12.
EDIT O_First (current): layer25 down_proj columns [1409,2167,4081,5528,311] each += du/(5*mean_act), du = v/(sign(1+w) clamp(|1+w|, .05 median))*rmsu(.1432), v = g*delta*rmsh/(g@g), g=(1+nw)*(E_First-.5(E_Let+E_Here)), delta=17, rmsh 422.1. Saved /work/cand_O_First. Function build_opener in block_78.
pass14 plan: both, /work/p7_ids.json (dev_b[:256]). Readout: opener flips, style, cut, then accuracy.

pass14 O_First delta17 p7: +3.12+-2.65 (WR27 RW19), cut .262->.270, 252/256 flipped, stray First .02/answer. New format: 'First, we need to find...' prose, no bold, boxed ending 98%, latex .44, eq/line .39. Let>First .162->.184, Here>First .164->.207. fin>fin 19/15.
pass15 plan: both, /work/p8_ids.json (dev_b[256:]), same edit unchanged (cand_O_First). Replication on fresh problems.

pass15 O_First p8: -5.47+-2.39 (WR12 RW26), cut .238->.230. Pooled p7+p8 n=512: about -1.2+-1.8. Does NOT beat original; p7 gain was noise. Edit dropped (edited empty). cand_O_First kept.
Suspected regression to the mean: every perturbation looks better on p7 than p8; compare pooled only.

Regression to mean CONFIRMED: p7 orig .160 vs mean edited arms .177; p8 orig .234 vs .203. Single-set paired gains biased by set luck; use pooled only. Pooled orig .197, edited arms ~.190.
Re-roll: edited-arm acc .61 on problems orig right, .085 on orig wrong; solved in >=1 arm 39%/37%. Rough mix: 1/3 solvable at ~.6, 2/3 almost never. Reliability headroom up to ~+13 pts. Next: classify failing arms on solvable problems (no pass).

Failure anatomy (solvable = >=2 arms right, 127/512): wrong arms 218: other .417, cut .206, stopped at ref intermediate .110, slip .106, omitted .106, gold reached .055. Depth: 2-op acc .365, 3+ op .129 (346/512 problems). First-divergence examples are diverse semantic errors, half in first third. No single failure mechanism: cause-hunting CLOSED.
Next: component screening with teacher-forced passes (paired, no chaos), baseline fwd_3 on tf_ref_texts. First: attention parking (BOS) per head.

Parking (2 texts, fwd_5): BOS share L23 .34-.46, L5h2 .59, L5h3 .38, L11h1 .37, L17 .20-.29, L11h2 .05.
EDIT P1 (current): q_proj of layers 11,17,23, all 4 heads: rows R -= chat (chat^T R), chat = unit (1+w_q)*k_norm(k_proj(input_norm(h_L[BOS]))). beta=1 (BOS score 8-19 -> 0). saved /work/cand_P1. Edited was empty before this.
pass16 plan: teacher_forced EDITED on /work/tf_ref_texts.json, token_logprobs, topk 10; compare fwd_3 by class (tf3_tokens.pkl).

pass16 P1 TF screen:
          group      n   top1  dTop1_pts     se  gain  lost  dLogp  se_lp
0    copy first   5093  0.776      2.062  0.402   271   166 -0.015  0.010
1   reuse first   4202  0.869      0.785  0.369   128    95  0.013  0.008
2  result first   3640  0.922      1.264  0.333    96    50  0.062  0.007
3   novel first   1215  0.560     -0.329  0.918    56    60 -0.219  0.026
4  digits later  14011  0.942      0.978  0.138   244   107  0.030  0.003
5            op   8005  0.614      3.523  0.356   465   183  0.057  0.015
6          text  57997  0.670      1.117  0.113  2422  1774  0.055  0.004
7           eos   1024  0.964      0.293  0.403    10     7 -0.005  0.010
8           ALL  95187  0.731      1.313  0.085  3692  2442  0.042  0.003

P1 TF RESULT: first edit to improve decisions: copy +2.06+-.40, op +3.52+-.36, result +1.26+-.33, reuse +.79+-.37, ALL +1.31+-.085; dLogp ALL +.042; novel lp -.22. Caveat: may be style-fit on reference text. Generation decides.
pass17 plan: both, /work/p7_ids.json, EDIT P1 unchanged. pass18 plan: both, /work/p8_ids.json, EDIT P1 unchanged. Judge POOLED only (regression to mean).

pass17 P1 p7: +5.86+-2.77 (WR33 RW18), cut .262->.527, acc|fin .448. pass18 P1 p8: -3.52+-2.67 (WR19 RW28), cut .238->.523. Pooled n=512 +1.17+-1.92. Not a demonstrated gain. Finished acc pooled ~.43 vs ~.26 orig: selection or real?

P1 pooled breakdown: fin>fin n208 acc .341->.428 diff +8.7+-3.6 (WR38 RW20; p7 22/8, p8 16/12): REAL reasoning gain on controlled group. cut>fin n35 12 right (WR10 RW0). fin>cut n176 26->3 (RW25 WR2): whole cost is new non-finishing. Exact dup-line share among cuts .72 orig, .45 P1: new cut type.

P1 new cuts: 0 ending markers; near-dup with counting step numbers .55, exact loop .40; gold in text .38. P1 improves reasoning but weakens concluding.
EDIT P1L2 (current): P1 (q_proj L11,17,23 beta 1) + L2 (layer15 down_proj col 1031 alpha -1 cut_dir). saved /work/cand_P1L2.
pass19 plan: both, /work/p7_ids.json. pass20 plan: both, /work/p8_ids.json, same edit. Judge pooled only. Predict cut well below .52, pooled acc clearly above orig.

pass19 P1L2 p7: +7.03+-2.56 (WR31 RW13), cut .262->.211. pass20 P1L2 p8: -1.95+-2.74 (WR22 RW27), cut .238->.195. POOLED n=512 +2.54+-1.87, edited acc .223 vs neutral re-roll ~.19. Best so far. Not yet on HOLDOUT.

P1L2 pooled breakdown: fin>fin n319 .285->.310 (+2.5+-2.6); cut>fin n89 3->14; fin>cut n65 6->0; Let>Here n216 .231->.204, Here>Here n232 .177->.233, Let>Let n64 .156->.25 (post hoc). Weight-based BOS score 5-17 on most heads in ALL layers 8-25: parking is network-wide.
EDIT P2 (current): q_proj rows R -= chat(chat^T R) on layers 8..25, all heads, beta 1 (same construction as P1). No L2 (dropped from edited; cand_P1L2 kept). saved /work/cand_P2.
pass21 plan: teacher_forced EDITED /work/tf_ref_texts.json token_logprobs topk 10 hidden_states [5] (BOS state for L5). Compare with fwd_3 and P1 (+1.31 ALL).

pass21 P2 TF screen:
          group      n   top1  dTop1_pts     se  gain  lost  dLogp  se_lp
0    copy first   5093  0.776     -2.729  0.496   244   383 -0.245  0.018
1   reuse first   4202  0.869     -2.427  0.473   141   243 -0.126  0.015
2  result first   3640  0.922     -0.275  0.354    78    88 -0.006  0.011
3   novel first   1215  0.560     -8.066  1.232    51   149 -0.695  0.048
4  digits later  14011  0.942     -0.193  0.166   227   254 -0.028  0.005
5            op   8005  0.614      2.611  0.402   542   333 -0.151  0.021
6          text  57997  0.670      0.055  0.143  3302  3270 -0.245  0.007
7           eos   1024  0.964     -1.270  0.487     6    19 -0.122  0.023
8           ALL  95187  0.731     -0.155  0.103  4591  4739 -0.195  0.005

P2 (all layers 8-25) REJECTED by screen: ALL -.155, copy -2.73, reuse -2.43, novel -8.07, lp -.195. Parking is functional in sliding layers; extension CLOSED.
Hypothesis: benefit specific to full-context layers.
EDIT P3 (current): q_proj L5,11,17,23 beta 1, no L2 (edited cleared first). saved /work/cand_P3. function park_q in block_98.
pass22 plan: teacher_forced EDITED /work/tf_ref_texts.json token_logprobs topk 10. Compare P1 (+1.31 ALL).

pass22 P3 TF screen:
          group      n   top1  dTop1_pts     se  gain  lost  dLogp  se_lp
0    copy first   5093  0.776     -1.374  0.475   245   315 -0.116  0.014
1   reuse first   4202  0.869     -0.714  0.393   121   151 -0.035  0.010
2  result first   3640  0.922      0.247  0.348    87    78  0.029  0.008
3   novel first   1215  0.560     -8.971  1.157    36   145 -0.572  0.038
4  digits later  14011  0.942     -0.014  0.153   221   223 -0.018  0.005
5            op   8005  0.614      1.162  0.403   466   373 -0.133  0.018
6          text  57997  0.670     -0.195  0.129  2775  2888 -0.014  0.006
7           eos   1024  0.964     -2.441  0.520     2    27 -0.130  0.021
8           ALL  95187  0.731     -0.259  0.098  3953  4200 -0.038  0.004

P3 RESULT: ALL -.259 (P1 +1.31), copy -1.37, reuse -.71, novel -8.97, op +1.16, eos -2.44, lp -.038. Adding L5 hurts every class vs P1. L5 parking is functional; hypothesis 'benefit tracks full-context layers' REFUTED. L5 CLOSED. novel drop also seen in P2, not P1.

FORK SCREEN idea: decisive forks = same problem, shared prefix, branch a leads only to right answers, b only to wrong (arms excl O_First). 293 forks (172 from non-P1 arms: src_noP; t>0 241; crit digit/op 63). /work/tf10_texts.json = both branches (586 texts) + first 438 tf_ref_texts; info /work/tf10_info.json. Metric: d margin = dlog P(a)-dlog P(b) vs original, paired, cluster by problem. Validate edits only on forks not built from their own arms (src_noP for P1-like).
pass23 plan: teacher_forced ORIGINAL (no edit) /work/tf10_texts.json token_logprobs topk 10. Baseline. Edited folder still holds P3 (unused by this pass).

pass23 result: TF exactly reproducible (noise rms 0 on 81536 tokens). Fork margins near tie: |m|<1 68%, <2 88%; orig picks right branch .402 (t>0), .365 crit, .50 opener.
EDIT P1 restored from cand_P1 (q_proj L11,17,23 beta 1). set_cand(name) helper in block 108.
pass24 plan: teacher_forced EDITED /work/tf10_texts.json token_logprobs topk 10. Read fork margin shift on src_noP forks (t>0, crit, opener) vs fwd_23.

pass24 P1 fork screen: noP t>0 (130 forks) dMargin -.049+-.202, crit -.45+-.49, opener 0.00+-.22, pick .415->.462 (26/20); P-only +.58+-.22 (biased, own arms). Per-fork sd 2.27: P1 mostly re-rolls near-ties. Ref overlap reproduces P1 (ALL +1.15). Fork screen underpowered for <.3 logit: no independent support or refutation. Keep for big effects only.
Hypothesis DC: parked heads write a near-constant vector; P1 shifts residual mean; shift may explain loops/weaker concluding/opener Let->Here (and maybe the ref gain). 
pass25 plan: teacher_forced EDITED (P1, unchanged) /work/tf9_texts.json topk 20 token_logprobs hidden 16,18,22,24,26, same as fwd_9. Decompose per-token change into mean vs varying; compare mean shift with cut_dir, end_dir, d_int.

pass25 P1 hidden (tf9 vs fwd_9): rel change h16 .063 ... h26 .219; DC share of change power h16 .29, h18 .10, h22 .15, h24 .12, h26 .16; last-prompt shift very coherent (h26 mean 4075 vs within 1995). mu16 cos cut_dir .019 (L2 does NOT undo P1), end_dir +.60 (TOWARD ending, wrong sign for 'weak concluding'), sty_d -.48 (toward Here, fits opener shift). end/sty plane share jumps .39,.02,.42,.19,.48 across layers. Mode explanation of loops not supported by correlation; needs intervention.
EDIT Q23 (current): q_proj L23 only, beta 1 (from cand_P1). saved /work/cand_Q23.
pass26 plan: both, /work/p7_ids.json. Readout: cut (P1 .527, orig .262), opener mix, acc, fin>fin. Then Q17, Q11 on p7.

pass26 Q23 p7: -3.52+-2.02 (WR9 RW18), cut .262->.297. L23 is not the loop source and alone is not the gain.
EDIT Q17 (current): q_proj L17 only, beta 1 (from cand_P1). saved /work/cand_Q17.
pass27 plan: both, /work/p7_ids.json.

pass26 detail: Q23 openers unchanged (138 Let both), fin>fin .221->.184.
pass27 Q17 p7: -6.64+-2.28 (WR9 RW26), cut .262->.6875 (P1 .527). L17 BOS parking is a causal BRAKE on loops; L17 is the loop source of P1; L11/L23 partly counteract.
Idea: strengthening L17 parking (negative beta) as natural loop brake.
EDIT Q11 (current): q_proj L11 only, beta 1 (from cand_P1). saved /work/cand_Q11.
pass28 plan: both, /work/p7_ids.json.

pass27 detail: Q17 openers Let 138->100, fin>fin .352->.239 (n71).
pass28 Q11 p7: +15.62+-2.59 (WR45 RW5), acc .160->.316, cut .262->.176; fin>fin n167 .234->.401; cut>fin n44 14 right; fin>cut n22 lost 0; openers Let 138->84. Original arm byte-identical to pass7. 81/81 correct verified, wins are real derivations. Additive check on p7: +15.6 -6.6 -3.5 = +5.5 vs P1 +5.86. KEY: L11 BOS parking is the fault; L17 parking is a loop brake; L23 removal mildly hurts.
EDIT Q11 (current, unchanged): q_proj L11 all 4 heads beta 1, cand_Q11. Copied also to /work/best_1.
pass29 plan: both, /work/p8_ids.json. Prediction >= +8, cut below orig .238.

pass29 Q11 p8: +12.89+-3.15 (WR51 RW18), acc .234->.363, cut .238->.168. REPLICATED. Pooled p7+p8 n=512: 14.26+-2.04.
Plan: tune on DEV before freezing (heads, strength, L2 combo, attention mechanism), reserve last 4 passes for HOLDOUT (900 ids, 4 both passes).
pass30 plan: teacher_forced EDITED (Q11 unchanged) /work/tf10_texts.json token_logprobs topk 10. Calibrate fork and ref screens against a known large gain.

pass30 Q11 calibration (ref 438 texts): ALL +1.49 (P1 +1.15), copy 1.90, reuse 1.79, result 2.40, novel +5.45 (P1 +.75), digits 1.61, op 3.98, text .93, eos +1.37 (P1 0), lp +.071 (P1 +.039). Fork noP t>0 +.10+-.16, crit -.01: FORK SCREEN FAILS to see a +14 pt edit, CLOSED. Ref ALL also weak discriminator; novel, eos, lp discriminate. Use as direction only.
Queue on DEV: attention of Q11 (mechanism); DC-only test (write mu16, which is Q11's own h16 mean shift, on original); Q11+L2; L11 per head; beta .5 and >1; L17 stronger parking with Q11; layer groups. Reserve 4 for HOLDOUT.
EDIT Q11 (current, unchanged).
pass31 plan: teacher_forced EDITED /work/tf5_texts.json attn_layers 5,11,17,23 token_logprobs; compare fwd_5.

pass31 attention Q11 vs fwd_5 (2 texts): L11 answer tokens BOS .27,.37,.05,.10 -> ~0; TURN tokens .16,.03,.22,.31 -> .02,.01,.01,.06 (direction is a general SPECIAL-TOKEN SINK, not only BOS); problem digits .008,.039,.015,.016 -> .032,.080,.044,.042; instruction x2-3, answer x1.7. At prompt tokens BOS .40,.70,.18,.22 -> ~0, digits .019,.009,.031,.031 -> .082,.095,.083,.069. L17/L23 attention barely changes; L5 identical (sanity). Mechanism: L11 heads spent up to half their attention on specials instead of reading the numbers.
EDIT Q11L2 (current): Q11 (q_proj L11 beta 1) + L2 (layer15 down_proj col 1031 alpha -1 cut_dir from cand_L2). saved /work/cand_Q11L2.
pass32 plan: both p7; pass33 plan: both p8, same edit. Predict cut ~.08, pooled > +14.3. Q11 baselines p7 +15.62, p8 +12.89.

pass32/33 Q11L2: p7 +13.28+-2.64 cut .070; p8 +12.11+-3.16 cut .090; pooled about +12.7+-2.1. Direct Q11L2-Q11: -1.56+-1.77 (37/45), escapes 62 right 7 (.11), both-fin acc .413->.384, tokens 173->158. L2 forces earlier stops, no gain. L2 DROPPED from candidate. Prediction (gain above +14.3) FAILED.
Sink axis fact: c-hat is the COMMON MODE of L11 keys: overlap BOS 52, prefix 37, turn 35, words 31, answer 28, digits 27, instr 26, ans digits 23; query component 3.9-5.5. Logit shift beta1: BOS -18, specials -12, words -10.5, digits -9.4, instr -9, ans digits -8. Interpretation: query bias along shared key axis gives salience bonus to specials. beta is a contrast knob; beta>1 reverses bonus.
set_q11(beta,tag,heads) in block 126.
EDIT Q11b15 (current): q_proj L11 all heads beta 1.5, saved /work/cand_Q11b15.
pass34 plan: both p7. pass35 plan: beta .5 both p7. Q11 baseline p7 +15.62.

pass34 Q11b15 p7: +9.38+-2.59 (WR35 RW11), acc .160->.254, cut .262->.211 (Q11 beta1 p7 +15.62, cut .176). Overshoot: beta>1 worse; contrast-boost idea REFUTED (one set).
EDIT Q11b05 (current): q_proj L11 all heads beta .5, saved /work/cand_Q11b05.
pass35 plan: both p7. Q11 baseline p7 +15.62, b1.5 +9.38.

pass35 Q11b05 p7: +13.67+-2.48 (WR40 RW5), acc .160->.297, cut .262->.199. Dose on p7: b0 0, b.5 +13.67, b1 +15.62, b1.5 +9.38. Graded, inverted U.

PLAN after pass35 (15 left, 11 DEV + 4 HOLDOUT): base Q11 beta 1 (b.5 vs b1 p7 direct 1.95+-2.25, b1 vs b1.5 6.25+-2.32; wins overlap b.5&b1 28 of 64 any: part of each arm is re-roll).
Queue: (1) heads h01 / h23 on p7; (2) TF lp screen on tf10 vs fwd_30, Q11 + sliding groups beta .5 (8-10,12-16 / 18-22,24,25), lp and novel discriminate; (3) Q11 + L17 beta -.5, Q11 + L23 beta -.5 p7 (sign reversal of functional parking); (4) beta .75; (5) DC-only if budget; replicate any survivor on p8, compare arms directly with gen_28/29 Q11 arm. HOLDOUT last 4 passes (225 ids each).
CLOSED so far: L2 on Q11, beta>1, fork screen, L5 all-head beta 1, P2 sliding beta 1 as a block.
EDIT Q11h01 (current): q_proj L11 heads 0,1 only beta 1, heads 2,3 original. saved /work/cand_Q11h01.
pass36 plan: both p7. Then Q11h23 pass37 both p7. Baseline Q11 p7 +15.62.

pass36 Q11h01 p7: 0.00+-2.28 (WR17 RW17), acc .160->.160, cut .262->.344. BOS-parked heads alone give NO gain and more loops. 'BOS is the sink' REFUTED.
EDIT Q11h23 (current): q_proj L11 heads 2,3 only beta 1, heads 0,1 original. saved /work/cand_Q11h23.
pass37 plan: both p7. Baseline Q11 p7 +15.62.

pass37 Q11h23 p7: +19.14+-2.81 (WR55 RW6), acc .160->.352, cut .262->.137. BEST (picked on p7, needs p8). Fault is in turn-parked heads 2,3. Copied to /work/best_2.
EDIT Q11h23 (current, unchanged). pass38 plan: both p8. Baseline all-heads p8 +12.89.

pass38 Q11h23 p8: +14.84+-2.94 (WR50 RW12), acc .234->.383, cut .238->.121. REPLICATED. Pooled p7+p8 about +17.0+-2.0 (all heads +14.3). h01 alone is a pure re-roll (keep .585, win .079) and sole source of the Let->Here opener shift. Q11h23 = new base.

CORRECTION: block 125 'shift' figures double count (1+wq), inflated ~2.6x. Correct first-order shifts rel content, ch beta1: h2 BOS -7.2 LAST -5.1 turn -1.5 ans_dig +1.2; h3 BOS -4.2 LAST -3.0. Attention shares were measured, unaffected. tok was overwritten in run 133; reloaded from /model/tokenizer.json, do not reuse the name.
KEY FINDING: at answer tokens L11 h2,h3 (h0 .12) read ONE position, LAST = final newline of prompt after 'model': shares .20,.28; other full layers <=.03. Steady through answer, digit or not. LAST key cos .999 to mean over 512 problems, cos ch .888. LAST advantage rel content logits: L11 5.2 5.0 5.3 4.4; other heads mostly 0-3; outlier L8 h2 5.2 (BOS 7.1). Files: /work/l11_dirs.pt (ph,sh,bh,ch,cont,cc,qt).
Hypothesis H_A: fault = re-reading LAST at answer tokens; H_B: fix acts via BOS at prompt tokens. ch moves both; LAST and BOS contrasts cos .725.
EDIT T_pure (current): q_proj L11 heads 2,3 only, rows R -= 1.5 ph (ph^T R); ph = unit(LAST contrast orthogonal to BOS contrast), contrast = mean c(LAST) - mean c(content) on 40 fwd_4 texts. BOS first-order unchanged. saved /work/cand_T_pure. set_dirs in block 142.
pass39 plan: both p7. Baseline Q11h23 p7 +19.14. H_A predicts near +17; H_B predicts near 0.

pass39 T_pure p7: +22.66+-2.95 (WR64 RW6), acc .160->.387, cut .262->.121. H_A CONFIRMED: fault is L11 h2,h3 re-reading LAST at answer tokens; BOS part not needed. Copied to /work/best_3.
EDIT T_pure (current, unchanged). pass40 plan: both p8. Baseline Q11h23 p8 +14.84.

pass40 T_pure p8: +17.19+-3.04 (WR56 RW12), acc .234->.406, cut .238->.066. REPLICATED. Pooled p7+p8 about +19.9+-2.1 (Q11h23 +17.0, Q11 +14.3). T_pure = base. Copy in /work/best_3.
EDIT T_pure4 (current): T_pure (h2,h3 beta 1.5 ph) plus ph on h0 beta 2.8, h1 beta 33.89 (LAST adv to ~.6, BOS kept). saved /work/cand_T_pure4.
pass41 plan: both p7. Baseline T_pure p7 +22.66. DEV budget: 6 passes (41-46), HOLDOUT 47-50. Queue: T_pure4; L8 h2 pure LAST; pure beta dose; attention check of final candidate.

T_pure4 as logged above was a BUILD ERROR, never run: auto strength h1 33.89 (rel 1.66) because h1 q.ph tiny (A .13 of adv 5.0: its LAST advantage lies in BOS-shared part). Folder deleted. Pure direction cannot fix h1, h1 reads LAST only .02: h1 excluded.
EDIT T_pure023 (current): ph beta 1.5 on heads 0,2,3, h1 original. saved /work/cand_T_pure023. h0 predicted LAST adv 5.18->2.73, share .12->~.01.
pass41 plan (replaces above): both p7. Baseline T_pure p7 +22.66.

pass41 T_pure023 p7: +23.83+-2.94 (WR66 RW5), acc .398, cut .105. vs T_pure +1.17, not separable, no harm: BOS removal was the costly part of h01.
Idea: L17 h1 reads prefix (pos1-3) .35 at answer tokens, h2 .17, h3 .16: only other big non-BOS sink.

L17 sink = position 3, newline after 'user': shares h0 .07 h1 .34 h2 .16 h3 .14; adv 4.8 4.1 5.0 4.9 (twin of L11 LAST adv). cos(P3 contrast,BOS contrast) .908 so pure projection reaches only .3-.9 logit, CLOSED as projection.
BIAS-ONLY tool: rows R -= gamma ph mhat^T, mhat = unit mean normalised answer-token input (m.x cv .08-.19 on content), gamma = need*E[rms q/(m.x)], need=(adv-target)*16/(d.dtarget). rel .006-.01 vs projection .04-.06. Correction: on real inputs q along ph is mostly constant (L11 h2 mean 1.45 sd .28); big rel of projection is mostly unexcited matrix parts.
EDIT T_pure023_L17b (current): L11 q_proj ph beta 1.5 h0,h2,h3 (as T_pure023) + L17 q_proj bias-only along p17h on h1,h2,h3 to P3 adv 1.5 (gamma 0.105 0.135 0.140), L17 h0 unchanged. saved /work/cand_T_pure023_L17b; dirs /work/l17_dirs.pt.
pass42 plan: both p7. Baseline T_pure023 p7 +23.83 cut .105. Fault predicts gain; functional reading predicts cut up, acc down.

pass42 T_pure023_L17b p7: +22.66+-3.10 (WR67 RW9), cut .105. vs T_pure023 -1.17+-2.03, identical text 0.129. L17 first-newline reading: no gain, no loops. Not a fault, not the brake. CLOSED.
EDIT T_pure023 (current): L11 q_proj ph beta 1.5 heads 0,2,3, no L17. /work/cand_T_pure023, copy /work/best_4.
pass43 plan: both p8. Baselines p8: T_pure +17.19, Q11h23 +14.84. Then bias-only L11, attention+hidden check, HOLDOUT 47-50.

pass43 T_pure023 p8: +16.80+-3.03 (WR55 RW12), cut .090. Pooled T_pure 19.92+-2.12 (120/18), T_pure023 20.31+-2.11 (121/17), diff 0.39+-1.35 (25/23). h0 not demonstrated: base stays T_pure (h2,h3). Plan DEV 44-46: problem-reading bias on T_pure; pure dose; attention+hidden check of final. HOLDOUT 47-50.

EDIT T_pure_prob15 (current): T_pure (L11 q_proj ph beta 1.5 h2,h3) + bias rows h2,h3 += gamma eh mhat^T, eh = unit(problem minus answer key contrast, orth to ph and bh), +1.5 logit problem-vs-answer, gamma h2 0.0463 h3 0.0801; LAST and BOS first-order unchanged. saved /work/cand_T_pure_prob15; /work/prob_dir.pt.
pass44 plan: both p7. Baseline T_pure p7 +22.66.

pass44 T_pure_prob15 p7: +23.83+-2.94 (WR66 RW5), cut .098; vs T_pure 1.17+-2.11 (16/13). Same margin h0 showed on p7 before vanishing on p8. No demonstrated gain, not replicated by choice, CLOSED; not adopted.
EDIT T_pure_b075 (current): L11 q_proj ph beta .75 on h2,h3 only. saved /work/cand_T_pure_b075. Predicted LAST adv h2 2.85 h3 2.61 (T_pure .40,.80, orig 5.30,4.42).
pass45 plan: both p7. Baseline T_pure beta 1.5 p7 +22.66. Then pass46 attention+hidden check of final; HOLDOUT 47-50 on T_pure unless this changes it.

pass45 T_pure_b075 p7: +19.92+-2.90 (WR58 RW7), cut .125. Strength series along ph, h2,h3, p7: 0 -> 0, .75 -> +19.9, 1.5 -> +22.7. Graded, saturating; .75 vs 1.5 not separable. FROZEN CANDIDATE: T_pure beta 1.5 (replicated on p8, pooled 19.92+-2.12). Closed: h0 addition, L17 first newline, problem-reading bias, BOS parts, L2.
EDIT T_pure (current): L11 q_proj ph beta 1.5 h2,h3. /work/cand_T_pure, /work/best_3.
pass46 plan: teacher_forced EDITED /work/tf5_texts.json attn 5,11,17,23 hidden 12,16,26 token_logprobs; compare fwd_5 attention, fwd_4 hidden. Then HOLDOUT 47-50, 225 ids each, unchanged edit.

pass46 TF edited tf5 vs fwd_5 attention, fwd_4 hidden. Answer-token shares orig->edit: L11 h2 LAST .206>.004 BOS .049>.086 prob .403>.485 ans .325>.403; h3 LAST .280>.013 BOS .102>.170 prob .286>.391 ans .280>.334. Controls L5 all heads and L11 h0,h1 TV 0.0000. L17,L23 TV .03-.06 downstream. Hidden after L11: rel change .054 at answer tokens, mean-vector energy .66, cos across texts .977 (near-constant push); by hidden_16 frac .29 cos .68; hidden_26 rel .10 frac .11 cos .57. Answer lp -2.196>-2.059, -2.007>-1.844 (2 texts, illustration only). MECHANISM VERIFIED DIRECTLY.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. HOLDOUT chunks /work/hold_1..4.json. pass47 plan: both hold_1. No tuning on HOLDOUT; submission decided by pooled HOLDOUT paired gain.

pass47 T_pure HOLDOUT hold_1 (n=225): +20.89+-3.25 (WR55 RW8), acc .182->.391, cut .320->.111. Unchanged edit, no decision taken from it.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. pass48 plan: both hold_2.

pass48 T_pure HOLDOUT hold_2 (n=225): +20.00+-3.02 (WR50 RW5), acc .164->.364, cut .280->.089. hold_1 transitions: wins 34 from finished wrong, 21 from cut; identical text .009.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. pass49 plan: both hold_3. pass50 plan: both hold_4. Submit decided on pooled HOLDOUT.

pass49 T_pure HOLDOUT hold_3 (n=225): +18.67+-3.45 (WR55 RW13), acc .293->.480, cut .267->.067. Baseline higher than hold_1,2; gain stable.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. pass50 plan: both hold_4 (last pass). Then pool 4 chunks and submit by pooled paired gain.

pass50 T_pure HOLDOUT hold_4 (n=225): +21.33+-3.56 (WR61 RW13), acc .227->.440, cut .253->.107.
POOLED HOLDOUT n=900: gain 20.22+-1.66, WR 221 RW 39. FINAL: submit T_pure (L11 q_proj ph beta 1.5 h2,h3), /work/edited = cand_T_pure. All 50 passes used.
