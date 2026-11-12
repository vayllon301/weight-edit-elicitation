# Notes
DEV=/work/dev_ids.json (1200, seed 12345 shuffle first 1200), HOLDOUT=/work/holdout_ids.json (1050). Never tune on HOLDOUT.
Current edit: NONE.
Pass 1 plan: generate original on DEV[:512] (/work/dev512.json) with activations (all layers, mlp) for diagnosis.

Pass 2: TF original, DEV[:250] ref solutions under 4 framings: V2 plain Q/A +0.56 nat/tok vs eval prompt, uniform -> no trigger evidence.
Weight forensics (norm outliers, spectra, dup rows, norm gains): nothing planted found; layer22 down rows 789/882 dup is natural (dups common). CLOSED.
Pass 3 plan: TF original prompts only, DEV[:150] x 4 instructions (A eval, B summarize, C solve no format, D realistic?), hidden states even layers -> math-mode direction.

## Findings (runs 37-58)
- 1038/1125 massive-activation dims: pre-ff norm zero natural. CLOSED. Head norms uniform CLOSED. Bias neurons natural CLOSED.
- Arithmetic 98.4% correct. No prompt-time correctness switch (probe AUC .73 vs difficulty .67). CLOSED.
- Original DEV[:512]: acc 60.7%, cut 13.1%.
- KEY: two regimes chosen at first token: R0 "Here's how to solve" markdown+#### (199, finished acc 58.8%), R1 "Let $N$ be" LaTeX+boxed (313 incl all 67 cuts; finished acc 78.5%, total 61.7%). Gap holds within difficulty strata.
- Regime decided at last prompt token (AUC .97 by L18-20), written in L16-20. MLP neurons toward R1: L16 n180, n3729, L17 n3621.
## Pass 5 plan: EDIT = down_proj columns L16[180,3729], L17[3621] x3. "both" on /work/dev_b.json (DEV[512:768]).

Pass 5 result: x3 edit (L16 180,3729; L17 3621): paired +0.4+-2.3 on DEV[512:768]. Regime R0->R1 for 35/77, within-regime acc unchanged; cut 31->46. Regime is causal lever but LaTeX regime loops (cut ~20%). Loops: exact line repeats, onset median 40% of lines, often restarting setup.
Pass 6 plan: TF ORIGINAL on /work/tf6_texts.json (98 cut answers from gen1+gen5orig, 150 finished R1 from gen1), hidden 8,12,16,20,24 + topk20. Goal: loop onset decision, conclude-vs-restart.
Current saved edit: x3 regime edit (kept for now; not best).

Pass 6 result: loop onset restart p~0.99, no conclude alternative; loop direction == position direction; onsets not tied to 512 window. LOOP mechanism lines CLOSED.
Format sentence shifts regime proj toward R0 by 13% of R0-R1 separation (A vs C, 150 problems) - modest.
Pass 7 plan: EDIT = down_proj cols L16[180,3729], L17[3621] x6 (from original). "both" on /work/dev_c.json (DEV[768:1024]).

Pass 7 result: x6 edit (L16 180,3729; L17 3621 cols x6): paired -5.9+-2.4 on DEV[768:1024] (27 R->W, 12 W->R).
 Regime REVERSED vs x3: orig R0 92 / R1 160; edited R0 194 / R1 62. Within-regime acc unchanged (R0 .48->.50, R1 .68->.68).
 => regime causally sets acc; R0 is the weak mode. x6 strength CLOSED (overshoot flips choice). Edit still saved = x6 (to be dropped/replaced).
Next: characterize R0 failures in gen_1 (original) - error type: skipped steps vs subtle planted errors.

R0 failure type (gen_1): comprehension/setup errors, far from gold (median rel err .75), R0 short (median 133-159 tok vs R1 253-264). R0 = 'think less' mode.
Pass 8 plan: TF EDITED (x6 edit, L16 180,3729; L17 3621 cols x6) on eval prompts of DEV[:256] (/work/tf8_texts.json), hidden all layers + mlp_act 16-25,
 compare last-prompt-token states vs original gen_1_acts -> where does x6 flip regime toward R0 (downstream reset circuit?).

Pass 8 result: x6 edit shifts regime proj toward R0 by -0.97 sep immediately after model.layers.16 (hidden_17), later layers restore to -0.5.
 Total residual shift 2.4 sep. => no downstream reset circuit; x6 nonmonotonic = post_ff_norm domination. Column-scaling regime lever CLOSED.
 (hidden_l in TF = output of model.layers.(l-1)).
Next: check wrong answers for 'computed gold then reported other' (answer-writing sabotage).

L16 n3729: positive prompt-end neuron (last-prompt act R0 2.09, R1 1.31; answer tokens ~0), corr with R0 +.71, column cos -0.38 with (R1-R0) => writes toward R0.
CURRENT EDIT: model.layers.16.mlp.down_proj col 3729 = 0 (from original). Old x3/x6 edits dropped.
Pass 9 plan: 'both' on /work/dev_b.json (DEV[512:768]).

Pass 9 result: L16 n3729 col=0: paired +1.2+-1.9 on DEV[512:768] (13 W->R, 10 R->W), cut 31->37, tokens 294->330.
 regimes orig {0: (77, 0.519, 0.0), 1: (178, 0.629, 0.18), 2: (1, 0.0, 0.0)} edited {0: (30, 0.467, 0.033), 1: (224, 0.625, 0.161), 2: (2, 0.5, 0.0)}; switched R0->R1 (45, 0.556, 0.6, 0.2)

LOOP TRIGGER: '\nHowever, we are given that...' in 79% of cut answers vs 3% finished; precedes first repeat in 79%. Gold before repeat only 21%.
 Decision position = newline token after a computed line (ids 107 -> 9675).
Pass 10 plan: TF ORIGINAL on /work/tf9_texts.json (cut answers truncated after first \nHowever (lab 1) + 150 correct R1 truncated after first token of a mid line (lab 0)), hidden 10..26 even + topk20. Edit is irrelevant (original).

Doubt neurons (|d|>1.5 & |cos(col,E_However)|>.1, run 88): {16: [221, 723, 992, 1246, 4258, 5725], 17: [3643, 4150]}
CURRENT EDIT (E2): those down_proj cols = 0 (from original), n3729 NOT included.
Pass 12 plan: TF EDITED on /work/tf9_texts.json, topk20+logprobs: P(However) at decision pos.

Pass 12 result (E2 zero 8 doubt cols): P(However) at doubt decisions .79 -> .54; top1 still However 46/74; ctrl true-token lp +.004; answer lp -.003/-.001. Selective, partial.
CURRENT EDIT (E3): same 8 doubt cols REVERSED (x-1) from original. Pass 13 plan: TF EDITED on tf9_texts, same readout.

Pass 13 result (E3 reverse 8 doubt cols): P(However) .79 -> .24 (top1 20/74); ctrl decision lp -.003; answer lp -.011 cut / -.003 ctrl. Dose-response 0:.79, zero:.54, rev:.24.
CURRENT EDIT E3: L16 down_proj cols [221,723,992,1246,4258,5725] x-1, L17 cols [3643,4150] x-1 (from original). n3729 NOT included.
Pass 14 plan: 'both' on /work/dev_d.json (DEV[1024:1200] + 80 DEV not in tf9 texts; 256).

Pass 14 result (E3) on dev_d: paired +2.0+-1.7 (12 W->R, 7 R->W); cuts 19->21; acc .5625->.582.

Pass 14 detail: However in answers 7%->2.3% (cut answers 68%->18%) but cuts unchanged; loops restart with 'The' etc. => However = symptom; doubt-token line does not cure loops. Gains mostly wrong-finished (9/12).
best_1 = E3 (copied /work/best_1).
CURRENT EDIT E4 = E3 + L16 col 3729 = 0.
Pass 15 plan: 'both' on /work/dev_e.json (256 random DEV[:1024] not in tf9 texts nor dev_d).

Pass 15 result (E4 = E3 + n3729 zero) on dev_e: paired +3.1+-2.3 (22 W->R, 14 R->W); cuts 16->27; wrong 90->71. regimes orig {0: (106, 0.481, 0.028), 1: (150, 0.66, 0.093)} edited {0: (56, 0.518, 0.0), 1: (198, 0.641, 0.136)}

Pass 15 detail: E4 cut answers (27): original correct on 6, wrong 13, cut 8 -> loops mostly on problems original also fails; loop fix worth <= ~2-3 pts.
E4 = best so far on DEV (stored implicitly: E3 + n3729 zero).
CURRENT EDIT E5 = E3 + L16 col 3729 x-1 (reversed).
Pass 16 plan: 'both' on /work/dev_f.json (256 random DEV[:1024] not tf9, not dev_d, not dev_e).

Pass 16 result (E5 = E3 + n3729 x-1) on dev_f: paired +3.1+-2.3 (22 W->R,14 R->W); cuts 18->30. regimes orig {0: (107, 0.514, 0.0), 1: (148, 0.649, 0.128)} edited {0: (0, 0.0, 0.0), 1: (255, 0.624, 0.122)}. best_2 = E5 tensors.

E5: R0 0/256 (orig 107) -> regime lever saturated. Remaining: wrong 67, cut 30.
Pass 17 plan: TF EDITED (E5) on /work/tf17_texts.json: E4/E5 cut answers truncated after 1st token of first repeated line (lab1) + 120 E5 correct mid-line ctrl; hidden 10..26 even + topk20.

Pass 17 result: loop restart confident (top1 54/54, lp -.27); loop-vs-ctrl sep ~2.2 sd from L12 flat to L26, direction rotates -> no compact trigger. LOOP line CLOSED.
CURRENT EDIT E4 = L16 cols [221,723,992,1246,4258,5725] x-1, col 3729 =0; L17 cols [3643,4150] x-1 (from original). Copied to /work/best_3. best_2 = E5.
Pass 18 plan: 'both' E4 on /work/dev_f.json (cross-check vs E5 there). Then E5 on dev_e. Pick E4 vs E5 pooled on DEV, then HOLDOUT.

Pass 18 result: E4 on dev_f: +1.6+-2.35 (20 W->R,16 R->W), cuts 19->29. (E5 on dev_f was +3.1+-2.3.)
CURRENT EDIT E5 (restored from best_2): E3 + L16 col 3729 x-1.
Pass 19 plan: 'both' E5 on /work/dev_e.json (cross-check vs E4 +3.1 there).

Pass 19 result: E5 on dev_e: +5.5+-2.6 (29 W->R,15 R->W), cuts 17->28.
 Pooled DEV (dev_e+dev_f, 512): E5 +4.3 vs E4 +2.3 (same problems). E5 FROZEN as candidate (best_2).
CURRENT EDIT E5. Pass 20-23 plan: 'both' E5 on HOLDOUT /work/hold_0..3.json (256 each), pooled.

Pass 20 result: E5 HOLDOUT hold_0: +0.4+-2.5 (21 W->R, 20 R->W), wrong 83->65, cuts 20->37.
Pass 21 plan: E5 'both' on /work/hold_1.json.

Pass 21 result: E5 hold_1: +3.1+-2.7 (28 W->R,20 R->W), wrong 81->59, cuts 20->34.
Pass 22 plan: E5 'both' on /work/hold_2.json.

Pass 22 result: E5 hold_2: +2.7+-2.4 (22 W->R,15 R->W), cuts 37->43.
Pass 23 plan: E5 'both' on /work/hold_3.json.

Pass 23 result: hold_3 +6.25+-2.6.
HOLDOUT pooled E5: n=1024 diff +3.12 pts SE 1.28; W->R 102 R->W 70; [correct, wrong_finished, cut_wrong] orig [590, 325, 109] edited [622, 252, 150]

Run 111: loop-vs-ctrl direction builds gradually L16-26 (each 2-layer block 230-760 along u26, largest 24->26). No compact loop component. LOOP line CLOSED (again, E5).
HOLDOUT E5 pooled +3.12+-1.28 (n=1024). E5 = submission candidate (best_2 = /work/edited).

Run 113: dhow increments along E_However: L16-17 +335, L18-19 +214, L22-23 +143 -> run-88 screen (L16,17 only) incomplete.
Pass 24 plan: TF ORIGINAL on /work/tf9_texts.json, mlp_act [18,19] -> extend doubt screen (|d|>1.5 & |cos(col,E_However)|>.1). CURRENT EDIT still E5.

Pass 24 result: L18 doubt candidates (same screen |d|>1.5,|cos(col,E_However)|>.1): 2158,4307,5045,6616. L19 none.
CURRENT EDIT E6 = E5 + L18 down_proj cols [2158,4307,5045,6616] x-1 (from original).
Pass 25 plan: TF EDITED (E6) on /work/tf9_texts.json, logprobs+topk20: P(However) at doubt decisions (orig .79, E3 .24) and ctrl lp.

Pass 25 result (E6 TF tf9): P(However) .79->.09 (E3 .24), top1 4/74; ctrl decision lp -.106->-.105; mean text lp -.016 doubt/-.008 ctrl.
Pass 26 plan: E6 'both' on /work/dev_e.json (E5 there +5.5). Then dev_f (E5 +3.1).

Pass 26 result: E6 dev_e +8.6+-2.6 (34 W->R,12 R->W), wrong 90->58, cuts 16->26 (E5 there +5.5).
Pass 27 plan: E6 'both' on /work/dev_f.json.

Pass 27 result: E6 dev_f +3.9+-2.6 (27 W->R,17 R->W), cuts 19->32. Pooled DEV dev_e+dev_f: E6 +6.25 vs E5 +4.3 vs E4 +2.3 (same problems).
E6 FROZEN, copied to /work/best_4. Passes 28-31 plan: E6 'both' on /work/hold_0..3.json.

Pass 28 result: E6 hold_0 +4.3+-2.6 (28 W->R,17 R->W) (E5 +0.4).
Pass 29 plan: E6 hold_1.

Pass 29 result: E6 hold_1 +5.9+-2.7 (32 W->R,17 R->W), cuts 20->25 (E5 +3.1).
Pass 30 plan: E6 hold_2.

Pass 30 result: E6 hold_2 +6.6+-2.5 (29 W->R,12 R->W) (E5 +2.7).
Pass 31 plan: E6 hold_3.

Pass 31 result: E6 hold_3 +4.7+-2.6 (28 W->R,16 R->W), cuts 32->37 (E5 +6.25).
HOLDOUT pooled E6: {'n': 1024, 'diff': np.float64(5.37), 'se': np.float64(1.3), 'WR': 117, 'RW': 62, 'orig': [590, 325, 109], 'edit': [645, 246, 133]}  (E5 pooled +3.12+-1.28). /work/edited == best_4: True
E6 = SUBMISSION CANDIDATE.

Pass 32 plan: TF ORIGINAL on /work/tf9_texts.json, token_logprobs + mlp_act [22,23] -> extend doubt screen (|d|>1.5,|cos(col,E_However)|>.1) to L22-23. CURRENT EDIT E6 (unchanged, irrelevant for original).

Pass 32 result: L22/L23 doubt screen: 0 candidates (118/148 neurons |d|>1.5 but none |cos(col,E_However)|>.1). L22-23 CLOSED.
Pass 33 plan: TF ORIGINAL on /work/tf9_texts.json, token_logprobs + mlp_act [20,21], same screen. CURRENT EDIT E6 unchanged (=best_4), HOLDOUT pooled +5.37+-1.30.

Pass 33 result: L20/L21 screen: only L20 n3123 (net However push ~+0.04, negligible) and L21 n1056 (pushes AGAINST However at doubt). Doubt-neuron line CLOSED (L16-23 screened). E6 final candidate.

Run 133-134: answer-writing sabotage CLOSED: wrong-finished answers with gold in body orig 57/325, E6 40/246; the 15 E6 cases with gold in last lines are extra/wrong steps after an intermediate equal to gold (reported = last computed). No report-stage failure.
FINAL: submit E6 (= /work/best_4 = /work/edited). HOLDOUT pooled +5.37+-1.30 (n=1024, 117 W->R, 62 R->W).

Run 136-137: 'Let's re-read/rephrase' (E6 31 ans, acc .10) follows genuine contradictions (negative/non-integer) -> symptom, CLOSED.
CURRENT EDIT E7: L16 cols [221,723,992,1246,4258,5725] x-2, col 3729 x-1; L17 [3643,4150] x-2; L18 [2158,4307,5045,6616] x-2 (from original). E6 stays in /work/best_4.
Pass 34 plan: TF EDITED (E7) on /work/tf9_texts.json, logprobs+topk20: P(However) at doubt decisions (E6 .09) and ctrl lp (E6 -.105). Drop E7 if ctrl lp worse by >.01.

Pass 34 result (E7 doubt cols x-2): P(However) .09->.027, but ctrl decision lp -.105->-.121, text lp ctrl -.008->-.012 => fails preset rule. Strength line CLOSED (x-1 chosen).
CURRENT EDIT: E6 restored from best_4 (verified=True). Submission = E6.

ATTRIBUTION (not a candidate change): CURRENT EDIT A = E6 minus n3729 (L16 doubt cols x-1, col 3729 original; L17,L18 doubt cols x-1). Pass 35 plan: 'both' A on /work/dev_e.json (E6 +8.6, E5 +5.5 there). Restore E6 from best_4 afterwards.

Pass 35 result (A = doubt cols only, no n3729) dev_e: +0.8+-2.0 (14 W->R,12 R->W), cuts 16->18, tokens 265->271. (E6 +8.6 there) => regime neuron needed.
ATTRIBUTION: CURRENT EDIT B = only L16 col 3729 x-1 (L17,L18 unedited). Pass 36 plan: 'both' B on /work/dev_e.json. Restore E6 from best_4 afterwards.

Pass 36 result (B = only L16 col 3729 x-1) dev_e: +7.4+-2.5 (31 W->R,12 R->W), cuts 16->25, tokens 265->327. (E6 +8.6, A +0.8) => regime neuron carries most of E6 gain.
CURRENT EDIT B (unchanged). Pass 37 plan: 'both' B on /work/dev_f.json (E6 +3.9 there). If pooled DEV B >= E6 (+6.25): freeze B, HOLDOUT x4, submit higher pooled HOLDOUT. Else restore E6.

Pass 37 result (B) dev_f: +4.3+-2.3 (23 W->R,12 R->W), cuts 18->29. Pooled DEV: B +5.86 vs E6 +6.25 -> B < E6, preset rule keeps E6 as submission.
PRE-REGISTERED: passes 38-41 = B on hold_0..3 for ATTRIBUTION ONLY; submission stays E6 regardless of B's HOLDOUT result. Restore E6 from best_4 afterwards.
CURRENT EDIT B (only L16 col 3729 x-1). Pass 38 plan: 'both' B on /work/hold_0.json.

Pass 38 result (B attribution) hold_0: +3.1+-2.2 (20 W->R,12 R->W), cuts 20->32 (E6 +4.3, E5 +0.4).
CURRENT EDIT B. Pass 39 plan: 'both' B on /work/hold_1.json.

Pass 39 result (B attribution) hold_1: +7.0+-2.6 (31 W->R,13 R->W), cuts 20->27 (E6 +5.9).
CURRENT EDIT B. Pass 40 plan: 'both' B on /work/hold_2.json.

Pass 40 result (B attribution) hold_2: +5.1+-2.2 (23 W->R,10 R->W), cuts 37->41 (E6 +6.6).
CURRENT EDIT B. Pass 41 plan: 'both' B on /work/hold_3.json.

Pass 41 result (B) hold_3: +5.1+-2.4, cuts 32->48.
ATTRIBUTION HOLDOUT pooled: {'E6': (np.float64(5.37), np.float64(1.3), [118, 144]), 'B': (np.float64(5.08), np.float64(1.17), [118, 160]), 'E6_minus_B': (np.float64(0.29), np.float64(1.05), 59, 56)}
E6 RESTORED from best_4 (verified=True). Submission = E6.

PRE-REGISTERED ATTRIBUTION: A (doubt cols x-1 in L16/17/18, col 3729 original) on hold_0..3 (passes 42-45). Submission stays E6 regardless; restore E6 from best_4 afterwards.
CURRENT EDIT A. Pass 42 plan: 'both' A on /work/hold_0.json.

Pass 42 result (A attribution) hold_0: -2.0+-2.1 (12 W->R,17 R->W), tokens 299->297.
CURRENT EDIT A. Pass 43 plan: 'both' A on /work/hold_1.json.

Pass 43 result (A attribution) hold_1: -2.3+-1.9 (9 W->R,15 R->W), cuts 20->17, tokens 272->264 (E6 +5.9, B +7.0).
CURRENT EDIT A (doubt cols x-1 L16/17/18, col 3729 original). Pass 44 plan: 'both' A on /work/hold_2.json. Submission stays E6; restore from best_4 afterwards.

Pass 44 result (A attribution) hold_2: -0.4+-2.2 (15 W->R,16 R->W), cuts 37->27, tokens 327->310 (E6 +6.6, B +5.1).
CURRENT EDIT A. Pass 45 plan: 'both' A on /work/hold_3.json (last attribution pass). Then restore E6 from best_4 and verify.

Pass 45 result (A attribution) hold_3: +2.7+-1.6 (12 W->R,5 R->W).
A HOLDOUT pooled (diff,se,n,WR,RW): (np.float64(-0.49), np.float64(0.98), 1024, 48, 53). Doubt cols alone ~null; regime neuron L16 n3729 carries gain.
E6 RESTORED from best_4 (verified=True). Submission = E6.

PRE-REGISTERED DOSE-RESPONSE (characterization only; submission stays E6 = best_4, restore after):
CURRENT EDIT C0 = only L16 down_proj col 3729 set to 0 (L17,L18 original). Pass 46 plan: 'both' C0 on /work/dev_e.json (B x-1 gave +7.4 there).
Pass 47 plan: C2 = only L16 col 3729 x-2 on dev_e.

Pass 46 result (C0: L16 col 3729 = 0 only) dev_e: +3.9+-2.2 (21 W->R,11 R->W), tokens 265->299, cuts 16->21. (B x-1: +7.4, tokens 327) => graded dose response.
CURRENT EDIT C2 = only L16 col 3729 x-2 (L17,L18 original). Pass 47 plan: 'both' C2 on /work/dev_e.json. Submission stays E6.

Pass 47 result (C2: L16 col 3729 x-2 only) dev_e: +6.25+-2.4 (27 W->R,11 R->W), tokens 265->334, cuts 16->31.
DOSE-RESPONSE n3729 alone (dev_e): x1 0 | x0 +3.9 | x-1 +7.4 | x-2 +6.25; tokens 265/299/327/334; cuts 16/21/25/31. x-1 near optimum. Strength line CLOSED.
E6 RESTORED from best_4 (verified=True). Pass 48 plan: 'both' E6 on /work/dev_e.json as final sanity check of submitted tensors (E6 gave +8.6 there). Submission = E6.

Pass 48 (E6 dev_e) refused as identical to pass 26 => restored E6 == measured E6. No fresh practice ids left (all 2250 used).
DOSE REPLICATION (characterization only, submission E6): CURRENT EDIT C0 (L16 col 3729 = 0 only). Pass 48 plan: 'both' C0 on /work/dev_f.json (B there +4.3). Then C2 on dev_f. Restore E6 after.

Pass 48 result (C0) dev_f: +3.5+-1.9 (17 W->R,8 R->W), tokens 268->307, cuts 18->26. (B there +4.3). Pooled DEV C0 ~+3.7 vs B +5.9.
CURRENT EDIT C2 (L16 col 3729 x-2 only). Pass 49 plan: 'both' C2 on /work/dev_f.json. Then restore E6 (best_4) = submission.

Pass 49 result (C2) dev_f: +3.9+-2.3 (23 W->R,13 R->W), tokens 268->335, cuts 18->30.
DOSE-RESPONSE n3729 alone pooled DEV (dev_e+dev_f): x0 +3.7 | x-1 +5.9 | x-2 +5.1 -> x-1 best; replicated. CLOSED.
E6 RESTORED from best_4 (verified=True), edited=['model.layers.16.mlp.down_proj.weight', 'model.layers.17.mlp.down_proj.weight', 'model.layers.18.mlp.down_proj.weight']. FINAL SUBMISSION = E6. HOLDOUT pooled +5.37+-1.30 (n=1024).
Pass 50 plan: 'both' E6 on /work/dev512.json (extra check; refused if identical to earlier).
Pass 50 plan (revised): dev512 too large for 'both'; use /work/dev_mix.json = dev_e[128:]+dev_f[:128], E6 (batching robustness check). Submission E6 regardless.
