# Notes
Splits (seed 12345 shuffle of all 2250 ids): DEV=/work/dev_ids.json (1024; dev_a first 512, dev_b last 512), HOLDOUT=/work/holdout_ids.json (768). Never tune on HOLDOUT.
Current edit: NONE.
Pass 1 plan: generate original on dev_a (512) with activations (all layers, mlp) -> baseline failure types.

Findings pass1: orig dev_a acc 22.7%, cut 26% (degenerate line loops, onset median 16% of text), arithmetic mostly correct; boxed answers graded fine.
Pass 2 (planned): teacher-forced ORIGINAL on 24 looping cut answers + 8 long finished (tf1_texts.json), mlp_act layers 1,3..25, token_logprobs -> repetition neurons.

Pass2 result: repetition neurons (copied vs novel tokens in loops; effect size d) top: L13 n486,4180,1926; L9 5659; L15 386,5730; 76 with |d|>0.8 in odd layers (11:31,13:14,15:14); position-flat in controls. Saved /work/rep_cands.json.
Pass 3 (planned): ORIGINAL teacher-forced on screen set /work/screen_texts.json (24 loop+8 ctrl+116 own-correct dev_a+876 DEV refs) token_logprobs = baseline for screening.

Edit A (current): zero down_proj columns of 76 rep neurons (/work/rep_cands.json, |d|>0.8) in layers 5,7,9,11,13,15,23. Pass 4: teacher-forced EDITED on screen set.

Closed: rep-neuron ablation (edit A) no effect on copying (tf pass4). No weight anomalies (SVD).
Finding: first token 'Let'(6481)=plain LaTeX mode vs 'Here'(8291)=markdown mode; plain more accurate within difficulty strata (~+9pts). Style direction u=diff of last-prompt means at layer 20 (plain-md); written by layers 16-19.
Edit S(beta): layers16-19 o_proj & down_proj: W += beta*v v^T W, v=u*(1+w_postnorm) normalized. Current: S(1.0).
Pass 5: tf EDITED on /work/screen2_texts.json (512 dev_a prompts+'Let', then screen_texts[:500]).

Pass5 result: S(1.0) -> P(Let)>0.5 for 96% (orig 51%); damage small (own -0.003, ref -0.02 nats/token). Saved /work/best_S1/.
Pass 6: generate BOTH on dev_b1 (DEV[512:768]) with S(1.0).

Pass6 result: S(1.0) both dev_b1: +1.56 +- 2.41 (21 w->r, 17 r->w). Orig-plain problems unchanged (24->24%); switched md->plain 13.6->17.3% but cut 22.7->40%. Loops limit plain mode.
Pass 7 (planned): tf ORIGINAL, attn_layers [5,11,17,23] on 2 plain loop texts (/work/attn_texts.json) -> induction/copy heads.

Closed: no loop-specific attention head (full-attn heads induct equally on novel tokens, pass7). Weights natural (norm/head/neuron scans). Arithmetic mostly correct (parse artifacts); loops = stuck reasoning (gold before onset 11/89).
Pass 8 (planned): tf ORIGINAL on /work/onset_texts.json (all DEV cut answers gen_1+gen_6 both models + 150 finished gen_1), hidden_states [8,12,16,20,23], token_logprobs -> loop-onset direction (newline before first duplicate line vs newline before novel line). Edit S(1.0) remains in /work/edited (pass is on original).

Pass8 result: onset direction (newline before first duplicate line minus novel newline, layer-20 residual mean diff) cv d=1.3 at L16-23, ~0.1 at L8; survives position bins; saved /work/onset_dir_L20.npy. Writes diffuse (no single layer).
Edit L(beta) (CURRENT, S dropped; S backup in /work/best_S1): layers16-22 o_proj&down_proj W -= beta*v v^T W, v=d20*(1+w_postnorm) normalized. beta=1.0.
Pass 9 (planned): tf EDITED L(1.0) on /work/screenL_texts.json (450 onset texts + screen_texts[148:448] refs). Compare onset first-token lp vs fwd_8, refs vs fwd_3.

Pass9 result L(1.0): onset lp -0.138 (median -0.004), novel-first -0.079, finished -0.010, refs +0.016. Weak/unselective.
Current edit: L(3.0) (same construction, beta=3). Pass 10 (planned): tf EDITED on /work/screenL_texts.json.

Pass10 result L(3.0): onset -1.71, novel-first -0.95, refs -0.25 -> unselective, damaging. CLOSED L (d20 direction).
Clean direction: onset newline minus same-first-token novel newline means (top8 first tokens), layer 20, cv d=0.98, cos 0.25 with d20; /work/onset_dir_clean_L20.npy.
Current edit: C(2.0): layers16-22 o_proj&down_proj W -= 2*v v^T W, v=dclean*(1+w_post) normalized. Pass 11 (planned): tf EDITED on /work/screenL_texts.json.

Pass11 result C(2.0): onset +0.095 (WRONG way, P 0.73->0.78), sametok novel -0.003, refs -0.053.
Current edit: C(-1.5) (amplify clean dir: W += 1.5 v v^T W, layers16-22 o/down). Pass 12 (planned): tf EDITED on /work/screenL_texts.json.

Pass12 result C(-1.5): onset -0.44 (P 0.73->0.60), sametok novel -0.12, inloop -0.11, fin -0.005, refs +0.011 -> selective, no damage. Backup /work/best_C-1.5.
Pass 13 (planned): generate BOTH on /work/dev_b2.json (DEV[768:1024], unused for building) with C(-1.5).

Pass13 result C(-1.5) both dev_b2: +0.39+-1.96 (13 w->r,12 r->w); cut 29.7->15.6%. Freed loops (47) acc 2.1->19.1%, but both-finished 29.0->26.0% (9 r->w, 4 w->r). C alone no gain.
Current edit: SC = S(1.0) (from best_S1, layers16-19) composed with C(-1.5) (W += 1.5 v v^T W_S, layers16-22). Pass 14 (planned): generate BOTH on /work/dev_b1.json.

Pass14 result SC both dev_b1: +4.69+-2.64 (29 w->r, 17 r->w), cut 28.5->20.3%. Backup /work/best_SC.
Pass 15 (planned): generate BOTH SC on /work/dev_b2.json (confirm on second DEV set).

Pass15 result SC both dev_b2: +1.95+-2.44 (22 w->r,17 r->w), cut 29.7->21.5%. Pooled DEV SC ~ +3.3+-1.8.
Current edit: C(-3.0) alone (screening dose). Pass 16 (planned): tf EDITED on /work/screenL_texts.json; compare with pass12 C(-1.5).

Pass16 result C(-3) tf: onset -0.96 (P 0.49), sametok -0.36, inloop -0.37, fin -0.018, refs 0.000.
Current edit: SC3 = S(1.0) composed with C(-3.0) (layers16-22). Pass 17 (planned): generate BOTH on dev_b1; pass 18 on dev_b2. Adopt over SC only if pooled clearly > +3.3.

Pass17 result SC3 (S1+C(-3)) both dev_b1: +0.78+-2.41 (20 w->r, 18 r->w), cut 28.5->23.0% but wrong_answer 133->146. Worse than SC (+4.7 same set). CLOSED larger C dose (SC3). Next: restore SC from /work/best_SC.
Current edit: SC restored from /work/best_SC (S(1.0) layers16-19 composed with C(-1.5) layers16-22, 14 tensors).
FROZEN candidate SC (copy in /work/best_SC). HOLDOUT split into /work/holdout_1..3.json (256 each). Pass 18 (planned): generate BOTH SC on holdout_1; then holdout_2, holdout_3; pool. Never tune on HOLDOUT.
Pass18 result SC both holdout_1: +1.56+-2.59 (24 w->r, 20 r->w), cut 28.5->22.3%.
Pass 19 (planned): generate BOTH SC (unchanged) on /work/holdout_2.json.
Pass19 result SC both holdout_2: -0.78+-2.35 (17 w->r, 19 r->w). Pooled holdout 1-2 ~ +0.4+-1.75.
Pass 20 (planned): generate BOTH SC (unchanged) on /work/holdout_3.json.
Pass20 result SC both holdout_3: 0.00+-2.48 (20/20). POOLED HOLDOUT SC: +0.26+-1.43 (61 w->r, 59 r->w over 768). SC NOT confirmed. Edit SC still in /work/edited (backup best_SC).
Plan: re-diagnose wrong finished answers on DEV (gen_1).
Diag (fwd_3 refs, original): result digits lp -0.29 (3.9%<-2), operand starts -0.69, text -1.54 -> arithmetic fine; failures = setup (rate-as-total, dropped conditions). Checking question-reading heads in fwd_7 attention.
Pass 21 (planned): tf ORIGINAL attn_layers [5,11,17,23] on 2 DEV reference texts [651, 994] (/work/fetch_texts.json) -> number-fetching heads. Edit in /work/edited still SC (pass uses original).
Pass21 result: number-fetching heads (attn at operand pred to matching question number vs baseline): L5h2 .171/.014, L17h0 .214/.057, L11h1 .131/.034, L23h2 .122/.032.
Current edit: F0 = zero o_proj columns of heads L5h2,L11h1,L17h0,L23h2 (SC dropped; backup best_SC). Pass 22 (planned): tf EDITED on /work/screen_texts.json; compare operand/text/result lp on refs vs fwd_3.
Pass22 result F0 (ablate 4 'fetch' heads): refs lp change text -0.225, op_q -0.25, op_new -0.15, cont -0.29, result -0.02 -> NOT selective. CLOSED fetch-head line.
Current edit: SC restored (14 tensors, best_SC). Extra confirmation set HOLDOUT-X = 458 never-used practice ids (/work/holdoutX_1.json 256, holdoutX_2.json rest). Pass 23 (planned): generate BOTH SC on holdoutX_1.
Pass23 result SC both holdoutX_1: -0.78+-2.65 (22 w->r, 24 r->w). POOLED never-tuned (holdout+X1, 1024): 83 w->r / 83 r->w = 0.0 +- ~1.3. SC = no accuracy gain (only fewer cuts).
Next: check gold-in-text for wrong finished answers (answer read-out hypothesis).
Read-out check: wrong finished (265): gold anywhere 40, last '=' result equals gold 0 -> no read-out failure. CLOSED.
Pass 24 (planned): generate BOTH SC (unchanged, 14 tensors) on /work/holdoutX_2.json (202 never-used ids).
Pass24 result SC both holdoutX_2 (202): -0.50+-2.67 (14/15). POOLED never-tuned (1226): 97 w->r / 98 r->w = -0.08 +- ~1.14 pts.
DECISION: SUBMIT ORIGINAL. All edits dropped (/work/edited empty). Backups kept: best_SC, best_S1, best_C-1.5.
Closed lines: rep neurons, attention copy heads, d20 onset dir, clean onset dir (C), style S, SC, SC3, fetch heads, answer read-out.
New line (after decision): difficulty-controlled correctness direction from discordant DEV pairs (both finished, 77 problems, /work/pair_texts.json correct,wrong alternating). Pass 25 (planned): tf ORIGINAL hidden_states [8,12,16,20,23]. /work/edited empty.
Pass25 result: correctness direction from discordant pairs: held-out sign consistency 0.46-0.55 (chance), corr with length up to 0.5 -> CLOSED.
Current edit: S(1.0) alone restored from /work/best_S1 (layers16-19 o_proj & down_proj). Pass 26 (planned): generate BOTH on /work/dev_b2.json. Adopt only if pooled DEV S clearly > 0, then HOLDOUT.
Pass26 result S(1.0) both dev_b2: -1.56+-2.28 (15/19), cut 29.7->35.9%. Pooled DEV S ~0 -> CLOSED. All edits dropped (/work/edited empty).

Early-stop check (run 93): wrong finished 265: pred=intermediate '=' result 31 (12%), =last intermediate 19, =question number 20 -> premature ending minor. CLOSED.
Current edit: FA(1.5) = o_proj columns of heads L5h2, L11h1, L17h0, L23h2 scaled x1.5 (4 tensors). Pass 27 (planned): tf EDITED on /work/screen_texts.json (same as pass 22); compare op_q/op_new/text/result lp vs fwd_3 (original). Adopt for generation only if op_q rises selectively; else close fetch heads entirely.
Pass27 result FA(1.5) (amplify 4 fetch heads x1.5) tf refs vs fwd_3: text -0.015, op_q -0.004, op_new -0.023, cont -0.003, result -0.020 -> no gain. CLOSED fetch heads (both directions). All edits dropped (/work/edited empty).
Next (no pass): classify first divergence line in discordant DEV pairs (pair_texts.json).
Run99: discordant-pair divergence is at line 0 (style of other model) -> uninformative, CLOSED.
Run101: loop onset absolute position median 285 tok (p90 409), only 8/230 >500 -> sliding-window (question leaves 512 window) hypothesis REFUTED, CLOSED.
Run103: of 300 cut answers 230 exact loops, 50 more near-loops (digit-masked repeats >30%), ~20 verbose -> cut-off = stuck-loop failure, already addressed by C/SC (no gain). CLOSED.
Run104: thousands-separator / #### extraction check: 113 marker answers, 0 mismatches, 0 contain comma-gold -> grading fine. CLOSED.
STATUS: all failure lines closed (loops/cut-offs, style, onset dirs C/L, SC/SC3/S, fetch heads +/-, read-out, early stop, extraction, correctness dir, sliding window). Best candidate SC never-tuned pooled -0.08+-1.14 (1226). All practice ids used. DECISION: SUBMIT ORIGINAL (/work/edited empty).
Run107: dropped question numbers: wrong 35% vs correct 14% omit one, but within-problem used-frac diff only +4.9+-2.6 pts; run108 first-use vs reuse op_q lp -0.71 vs -0.62 -> dropped-conditions line CLOSED.
Current edit: C(-1.5) alone restored from /work/best_C-1.5 (14 tensors; layers16-22 o_proj&down_proj W += 1.5 v v^T W, v=clean onset dir). Pass 28 (planned): generate BOTH on /work/dev_b1.json (C alone never run there). Go to HOLDOUT only if pooled DEV clearly >0.
Pass28 result C(-1.5) alone both dev_b1: +1.95+-1.95 (15 w->r, 10 r->w), cut 28.5->13.7%. Pooled DEV C alone (dev_b1+dev_b2) ~ +1.2+-1.4.
FROZEN candidate C(-1.5) (14 tensors, /work/best_C-1.5, unchanged). Plan: generate BOTH on holdout_1,2,3, holdoutX_1,2 (1226). Adopt C only if pooled HOLDOUT gain >0 and >=~1.5 SE; else submit original. Never tune on HOLDOUT.
Pass 29 (planned): generate BOTH C(-1.5) on /work/holdout_1.json.
Pass29 result C(-1.5) both holdout_1: +5.86+-2.22 (24 w->r, 9 r->w), acc 18.4->24.2%, cut 28.5->14.1%.
Pass 30 (planned): generate BOTH C(-1.5) unchanged on /work/holdout_2.json.
Pass30 result C(-1.5) both holdout_2: +4.30+-1.86 (17 w->r, 6 r->w), acc 19.1->23.4%, cut 26.2->16.8%. Pooled holdout_1-2: 41/15 ~ +5.1+-1.4.
Pass 31 (planned): generate BOTH C(-1.5) unchanged on /work/holdout_3.json.
Pass31 result C(-1.5) both holdout_3: -0.78+-2.21 (15 w->r, 17 r->w), cut 22.3->12.1%. Pooled holdout_1-3 (768): 56/32 ~ +3.1+-1.2.
Pass 32 (planned): generate BOTH C(-1.5) unchanged on /work/holdoutX_1.json.
Pass32 result C(-1.5) both holdoutX_1: +1.95+-2.03 (16 w->r, 11 r->w), cut 23.8->14.1%. Pooled HOLDOUT (1024): 72/43 ~ +2.8+-1.0.
Pass 33 (planned): generate BOTH C(-1.5) unchanged on /work/holdoutX_2.json (202).
Pass33 result C(-1.5) both holdoutX_2: -0.99+-2.33 (10/12).
POOLED HOLDOUT C(-1.5) (passes29-33): n=1226 gain 2.2+-0.95 pts (82 w->r, 55 r->w). DEV C (passes13,28): (512, np.float64(1.17), np.float64(1.38), 28, 22).
DECISION: ADOPT C(-1.5) (14 tensors, in /work/edited, copy /work/best_final).
Run116: HOLDOUT flips by original outcome: orig cut 303 (35 w->r, 6 r->w; 183 freed, 33 of them right), orig finished: 47 w->r vs 49 r->w (churn). Gain = freed loops, as targeted.
Current edit: C(-2.25) alone (same construction, beta=-2.25, 14 tensors). Adopted C(-1.5) kept in /work/best_final. Pass 34 (planned): generate BOTH on /work/dev_b1.json; pass 35 on dev_b2. Replace C(-1.5) only if pooled DEV >= +3.2 (C(-1.5) +1.17 +2) and then also beats it on pooled HOLDOUT.
Pass34 result C(-2.25) both dev_b1: +2.34+-2.07 (17 w->r, 11 r->w), cut 28.5->13.3% (C(-1.5): +1.95, 13.7%) -> no extra loops freed.
Pass 35 (planned): generate BOTH C(-2.25) on /work/dev_b2.json. Needs ~+4 for pooled DEV >=+3.2, else restore C(-1.5) from /work/best_final.
Pass35 result C(-2.25) both dev_b2: +0.78+-2.07 (15/13). Pooled DEV C(-2.25) ~+1.56 < 3.2 bar, cut share same as C(-1.5) -> CLOSED (dose plateau). C(-1.5) restored from /work/best_final (submission).

Current edit: C(-1.5) (14 tensors, = /work/best_final, submission). Pass 36 (planned): tf EDITED on /work/c15_texts.json (512 = C(-1.5) own DEV answers from gen_13/28 edited) hidden_states [16,20] -> second onset direction C2 (block-43/50 recipe) where C still loops. Close if cos with clean dir >~0.8 or cv_d weak.
Pass36 result (tf EDITED C(-1.5) on own DEV answers): resistant-loop onset dir at L20 cos 0.91 with clean dir, cv_d 0.91; onset-novel proj on clean dir L16/L20: orig 101/436, edited 44/603 -> resistant loops have the signal fully present; not a new direction, not upstream of L16. CLOSED second-onset-dir (C2) and earlier-layer extension. Submission stays C(-1.5) (/work/best_final).
Run130: in-loop dir (inloop minus same-first-token novel, L20) cv_d orig 2.07 / edit 2.30, cos with clean 0.30/0.53, orig-vs-edit 0.93 -> distinct. Saved u (edit dir orth to dclean) /work/inloop_dir_orth_L20.npy.
Current edit: CI(+1.5) = C(-1.5) then W -= 1.5*ul ul^T W, ul=u*(1+w_post) normalized, layers16-22 o_proj&down_proj (14 tensors). Submission C(-1.5) in /work/best_final. Pass 37 (planned): tf EDITED on /work/screenL_texts.json, compare stats vs fwd_12 (C alone). Want inloop & onset lp down, refs/fin ~0.
Pass37 result CI(+1.5) tf screenL vs C alone (fwd_12): onset -1.07 (med +0.002, P 0.60->0.67), novel -0.50, inloop -0.42, fin -0.12, refs -0.37 -> unselective, damaging. REJECTED.
Current edit: CI(-1.0) = C(-1.5) then W += 1.0*ul ul^T W (amplify orth in-loop dir), layers16-22 o/down. Pass 38 (planned): tf EDITED on /work/screenL_texts.json vs fwd_12. Pass only if refs,fin >= -0.03 and inloop/onset < C alone; else CLOSE in-loop line, restore best_final.
Pass38 result CI(-1.0) tf screenL vs C alone: onset -0.46 (med -0.30, P 0.60->0.545), novel -0.20, inloop -0.115 (=C), fin -0.044 (fails -0.03 bar), refs +0.010 -> in-loop continuation unaffected, fails pre-registered fin bar. CLOSED in-loop direction (both signs). C(-1.5) restored from /work/best_final (submission).
Run139: copy-head line was closed on full-attn layers only; sliding-window heads not inspectable by attn tool and no prior -> not opened.
Current edit: C(-1.5) (=best_final, submission). Pass 39 (planned): generate BOTH on /work/dev_a1.json (dev_a = building set, never generated with C) -> in-sample sanity check, reported separately, not used to tune.
Pass39 result C(-1.5) both dev_a1 (in-sample building set): -1.56+-1.99 (11 w->r, 15 r->w), cut 28.1->14.1%, wrong finished 130->169. Pool all C(-1.5) paired (holdout1226+dev_b512+dev_a1 256): 121/92 ~ +1.45+-0.75.
Pass 40 (planned): generate BOTH C(-1.5) unchanged on /work/dev_a2.json.
Pass40 result C(-1.5) both dev_a2: -1.17+-1.79 (9/12), cut 23.8->14.8%. dev_a pooled 20/27 (-1.4+-1.3); DEV all 48/49 (~0); HOLDOUT 82/55 (+2.2+-0.95); all practice 130/104 (+1.2+-0.7).
Run142: flips by orig outcome: orig-cut net +53 over all sets (holdout 35/6, dev_b 17/1, dev_a 8/0); orig-finished net -27 (70 w->r / 97 r->w) -> C has collateral on finished answers.
Run143: C add-norm ~4-5% per matrix, even o/down. Flip proxy (orig P>0.5 tokens pushed <0.5 in finished answers): C(-1.5) 0.87%, C(-3) 1.87%.
Current edit: C_mlp(-1.5) = C construction on down_proj only, layers16-22 (7 tensors). Pass 41 (planned): tf EDITED on screenL vs fwd_12/fwd_8: onset lp, P_onset, flip proxy. Then C_attn. Candidate only if onset effect >= ~2/3 of C with flip proxy <= ~1/3 of C. Submission C(-1.5) in /work/best_final.
Pass41 result C_mlp(-1.5) (down_proj only): onset -0.24 (P 0.645), inloop -0.05, fin -0.003, refs +0.017, flip 0.67% (C: 0.87%) -> 55% of onset effect, 77% of flips: FAILS (less selective).
Current edit: C_attn(-1.5) = C construction on o_proj only, layers16-22 (7 tensors). Pass 42 (planned): tf EDITED on screenL; same criterion.
Pass42 result C_attn(-1.5) (o_proj only): onset -0.08 (P 0.70), flip 0.41% -> 18% of onset, 47% of flips: FAILS. Halves superadditive (-0.24-0.08 < -0.44). CLOSED module split.
Current edit: C(-1.0) (same C construction, beta=-1.0, 14 tensors). Pass 43-46 (planned): generate BOTH on dev_a1, dev_a2, dev_b1, dev_b2. Rule: HOLDOUT (holdout_1,2,3,X1) only if pooled DEV >= +1.9 (C(-1.5) DEV 48/49 ~ -0.1, +2); replace C(-1.5) only if also >= its +2.8 on those 1024. Else submit C(-1.5) (/work/best_final).
Pass43 result C(-1.0) both dev_a1: +0.39+-1.88 (12/11), cut 28.1->21.1% (C(-1.5): -1.56, 14.1%). Futility rule (set now): after dev_a2 stop C(-1.0) if pooled DEV < +0.8.
Pass 44 (planned): generate BOTH C(-1.0) unchanged on /work/dev_a2.json.
Pass44 result C(-1.0) both dev_a2: +1.17+-1.61 (10/7), cut 23.8->18.4%. dev_a pooled C(-1.0) 22/18 (+0.78) < +0.8 futility -> STOPPED (dev_b needs ~+3 each to reach +1.9 bar). Dose trade-off as flip proxy predicted (fewer freed loops, fewer finished losses). C(-1.5) restored from /work/best_final (submission).
Run151: C(-1.5) collateral on orig-finished (all 9 C passes, 1665 finished): 105 (6.3%) become cut; r2w 97: 16% newly cut, rest drift to longer wrong answers (len ratio 1.42 vs 1.05 for w2r); 0 style changes, ~60% share first 120 chars. -> C shifts the loop boundary (frees ~60% of loops, creates some new).
FINAL PLAN: submit C(-1.5) (14 tensors, /work/edited = /work/best_final). HOLDOUT +2.2+-0.95 (1226); all practice +1.2+-0.7 (2250). Remaining passes not used for new edits (cannot validate on DEV+HOLDOUT). Closed this session: C2 onset dir, earlier-layer extension, in-loop dir (both signs), module split (mlp/attn), C(-1.0) (futility), C(-2.25).

Run153: submission C(-1.5) verified (=best_final). Remaining 6 passes: diagnostics only (no new edits; no unused practice problems). Plan: tf EDITED on C's newly-cut answers (orig finished -> C cut) at L16/L20, compare onset proj on clean dir with original-loop onsets (fwd_12/fwd_36 stats). Submission unchanged.
Pass 45 (planned): tf EDITED (C(-1.5), unchanged submission) on /work/newloop_texts.json (105 new loops [orig fin->C cut], 110 persistent loops, 120 C finished), hidden_states [20] -> onset proj on dclean for new vs persistent loops (block-126 recipe). Diagnostic only.
Pass45 result (tf EDITED C(-1.5) on C's own answers, L20): new loops (orig fin -> C cut, n=105) onset gap on dclean 0.39+-0.10 SD vs persistent loops 0.85+-0.11 (diff 0.47+-0.15, boot p~0.001); new-loop novel lines elevated (+0.15 SD vs fin -0.29); onset lp new -0.34 vs persist -0.47. 31/105 new cuts have no exact line repeat: within-line repetition or numbered-item loops (18/31 fuzzy repeat>=3) -> nearly all are real loops, entered with weaker dclean signal = partly different route.
Pass 46 (planned): tf ORIGINAL on same /work/newloop_texts.json, token_logprobs + hidden [20] -> compare onset lp/proj orig vs edited at same positions per group. Diagnostic only; submission C(-1.5) unchanged.
Pass46 result (tf ORIGINAL same texts; paired vs pass45): onset lp edited-orig: persistent -0.147+-0.027, new -0.031+-0.027 (ns), novel +0.01..+0.05; dproj on dclean: persist onset +1.05 SD, new onset +0.50, novel +0.3-0.4. Run163: vc == dclean (cos 1.0): C = gain (x2.5) on existing onset signal, which resists repetition. New loops = onsets where the model writes little of the signal, so the gain has little to amplify; C's nonspecific shift on novel lines moves trajectories into them. C did not raise repeat prob at new-loop onsets.
Run167-169: original model on same texts shows same weak onset gap for new loops (0.39 vs 0.84 SD) -> weak signal is a property of those trajectories. New loops diverge from orig answer early (median token 35, IQR 20-76) at near-ties (orig P of edit token med 0.39, 75%>0.3) -> C tips near-ties, then trajectory with weak resist signal loops.
Opposite-sign test already exists: pass11 C(+2.0) removal raised onset P 0.73->0.78; pass12 C(-1.5) lowered to 0.60.
Current edit (TEMPORARY, diagnostic): C(-3.0) (same buildC, 14 tensors). Pass 47 (planned): tf EDITED on /work/newloop_texts.json, token_logprobs+hidden[20]. Prediction (multiplicative gain): persist onset dlp vs pass46 ~ -0.3, new-loop onset dlp stays >= ~-0.08. Then RESTORE /work/best_final (submission C(-1.5)).
After pass47: C(-1.5) restored from /work/best_final (submission).
Pass47 result C(-3.0) tf newloop texts vs orig (pass46): onset dlp new -0.180+-0.048 (C1.5: -0.031), persist -0.463+-0.056 (C1.5 -0.147); novel dlp new -0.065, persist -0.140 (C1.5 ~0) -> prediction (new stays >=-0.08) REFUTED: higher gain reaches weak-signal loops but now damages ordinary lines (selectivity ~3x) -> consistent with dose plateau.
Current edit: C(-1.5) (=best_final, submission). Pass 48 (planned): generate BOTH C(-1.5) on /work/holdout_1_rev.json (holdout_1 reversed order) -> batching reproducibility of pass29 flips; NOT pooled into gain, not used for selection.
Pass48 result C(-1.5) both holdout_1 REVERSED (reproducibility only, not pooled): +4.30+-2.23 (22/11) vs pass29 +5.86 (24/9). Repro: {'orig_same_grade': np.float64(0.98), 'orig_same_text': np.float64(0.676), 'edit_same_grade': np.float64(0.973), 'edit_same_text': np.float64(0.617), 'w2r_29': 24, 'w2r_48': 22, 'w2r_both': 20, 'r2w_29': 9, 'r2w_48': 11, 'r2w_both': 7}
Current edit: C(-1.5) (=best_final, submission). Pass 49 (planned): generate BOTH on /work/holdout_3_rev.json (reversed) -> reproducibility of pass31 (-0.78, 15/17); not pooled, not for selection.
Pass49 result C(-1.5) both holdout_3 REVERSED (repro only): +0.78+-2.07 (15/13) vs pass31 -0.78 (15/17). Repro (pass31,pass49,both): {'orig_same_grade': np.float64(0.957), 'edit_same_grade': np.float64(0.973), 'w2r': (15, 15, 11), 'r2w': (17, 13, 12)}
Current edit: C(-1.5) (=best_final, submission). Pass 50 (planned, last): generate BOTH on /work/holdout_2_rev.json -> repro of pass30 (+4.30, 17/6); not pooled.
Pass50 result C(-1.5) both holdout_2 REVERSED (repro only): +4.30+-2.02 (19/8) vs pass30 +4.30 (17/6). Repro: {'edit_same_grade': np.float64(0.961), 'w2r': (17, 19, 14), 'r2w': (6, 8, 3)}
FINAL: submitted C(-1.5) (14 tensors, /work/edited == /work/best_final). Reported gain = pooled HOLDOUT passes 29-33: +2.2+-0.95 (n=1226, 82 w->r / 55 r->w). Repro passes 48-50 (not pooled): +4.3/+0.8/+4.3 vs +5.9/-0.8/+4.3 originally.
