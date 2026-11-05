# NOTES
Split: seed 20261105; /work/dev_ids.json (1500), /work/holdout_ids.json (750). HOLDOUT only for confirming a frozen candidate.
Current edit: NONE.
Plan: pass 1 = baseline generation of original on DEV[:512] (/work/dev_512a.json).
Closed lines: none.

Pass 2 (teacher-forced, ORIGINAL, no edit): /work/tf_frames_texts.json = 150 DEV reference solutions x 5 framings + 137 arithmetic probes x 2 (chat/raw). Purpose: is digit computation gated by context? Index: /work/tf_frames_index.json.
Findings so far: baseline DEV512 20.9% (107/512), 289 wrong, 116 cut; gold absent from 246/289 wrong; cut answers are loops; format is not the gap.

Closed: gate/up per-neuron balance (natural-looking main group + smooth tail, outliers carry 3-6% energy); RMSNorm weights smooth; raw arithmetic weakness (chat-only eval, not relevant); format gate on digits (flat across framings). Caveat: 21% may be near natural for a 1B model, implant NOT assumed.
Pass 3 (teacher-forced, ORIGINAL, no edit): /work/tf_contrast_texts.json, index tf_contrast_index.json. Groups M_correct, M_wrong, T_summ, F_fact, R_logic. Hidden states 0-26. Purpose: health check and layer where math vs other diverges.

Closed: correct-vs-wrong neuron correlations (hits at layer 0, last prompt token = difficulty confound). General health OK (facts fine, misses are style such as bolding). No sharp math gate in stream at last prompt token (pass 3). Layer 25 up_proj top dirs diffuse (275 neurons), not aligned with mean input.
Pass 4 (both, activations, DEV dev_256a.json = DEV[:256]): EDIT = model.layers.25.mlp.up_proj.weight minus its top 2 singular components (full removal). Purpose: causal test of the layer 25 anomaly + collect wrong/right flip pairs on same problem.
Plan: use flips as within-problem contrast for directions; profile sub-skills (binding, op choice, multi-step) by TF if needed.

Pass 4 result: removing L25 up_proj top-2 singular comps: acc 18.4 -> 1.2 (-17.2 +/- 2.6), 100% cut, immediate repetition, arithmetic in lines fine. So this rank-2 input channel = REPETITION BRAKE (natural machinery, needed).
Pass 5 (both, activations, dev_256a.json): EDIT = same tensor, top-2 singular values x1.4 (amplify). Purpose: dose-response on loops and accuracy.

Pass 5 result: L25 up_proj top-2 x1.4: acc -1.2 +/- 2.1, cut 22.7 -> 32.8. Both directions raise loops, so original sits near optimum. CLOSED: L25 top-2. Lead left: L25 direction 2 (s=5.97) proj differs cut vs finished (t=11, observational).
Closed: L23 o_proj spectrum (tail normal, earlier effective-rank impression was an artefact).
Pass 6 (teacher-forced, ORIGINAL, no edit): attention patterns layers 5,11,17,23 on 2 texts (/work/tf_attn_texts.json, index tf_attn_index.json). Purpose: is L23 attention diffuse (k_proj imbalance: trace +30, top eig 41)?

Closed (pass 6 + runs 30-32): L23 attention not diffuse; k/v extra direction not aligned with <bos> (cos 0.055); gate/up/down pairing normal, no dead neurons, kurtosis no break at L23-25; high down-vs-up z neurons are natural tail (normal column size, <=2.6% output, no shared direction). WEIGHT-BASED DETECTION EXHAUSTED: no implant found, none proved absent.
Behaviour: k=1 acc 27%, k=2 36%; per-step reliability about 0.6 for all ops; failures start at step 1; overshoot real (gold in 15% of wrong answers, ceiling about +8 pts); loops 23%.
Pass 7 (teacher-forced, ORIGINAL, no edit): /work/tf_screen_texts.json (index tf_screen_index.json). C=own correct, W=own wrong finished, R=reference for same problems. Hidden 8,12,16,20,24,26. Baseline for TF screen; R vs W is within-problem contrast.
Plan: TF-screen candidates (score: dLP(C)-dLP(W), plus dLP(R)); test: repetition contrast (same line first vs repeated) for loops; steering from R vs W at mid layers; head scaling for digit-reading heads; L25 dir 2.

Runs 34-35: difference-in-differences direction (own-ref on right minus own-ref on wrong problems): split cos 0.45-0.65 vs null 0; out of sample weak (0.13-0.22 text sd), not token composition, strongest in first 20 tokens. May still reflect difficulty. Needs causal test.
Pass 8 (teacher-forced, ORIGINAL, no edit): per-token mlp_act layers 7,11,15,19,23 on 60 own answers (/work/tf_mlp_texts.json, index tf_mlp_index.json). Purpose: find token-level near-constant writer neurons for a steering write.

Run 37: per-token cv best 0.23-0.38, no neuron under 0.2 (answer-level "near-constant" overstated).
Pass 9 (teacher-forced, EDITED): model.layers.15.mlp.down_proj.weight col 1031 += gamma*DiD(h16)*rms_y/w / mean_act, gamma=3.0, dims |w|>=5, rms_y=0.080 (estimate from mean act). Texts tf_screen_texts.json, hidden 16,20. Purpose: calibrate write size; score dLP(C)-dLP(W), dLP(R) vs base_lp7.npy.


Runs 42-48 (no pass used): flip reproducibility at h12/h16 survives disjoint problems (cos 0.86/0.67) but is the STYLE MODE. All 512 baseline answers open 'Here's...' (H, 259, bold steps) or 'Let...' (L, 253, LaTeX). PC0 of h12 mean-answer state (57% var) = mode. Acc by steps H vs L: k2 .32/.44, k3 .09/.25, k4 .11/.15, k5 .09/.10. Finished acc H .218 L .318; cut H .166 L .304. Pass 10 steering = mode push (20 H->L, 0 L->H; those 21 problems 0->2 correct).
REFUTED: 'H loops restart from opener' (tie-break artefact of most_common; no cut answer has any marker).
Headings: 'calculate the total' 37% of H headings; H answers with fewer steps than k acc 4%, equal 33%, more 22%.

Closed run 52: outsized near-constant neurons (L23 #5826, L16 #3437, L10 #1868/768, L22 #5108/4909) are bias units: constant across positions, sparse columns, mirror pairs add rather than cancel. H first heading 'calculate the total' acc 13% vs other 24%, matched by k (k2 22/44, k3 3/15), may be selection.
Pass 11 (both, activations layers 13,16,20,26, /work/dev_256c.json = DEV[512:768], fresh): EDIT = model.layers.15.mlp.down_proj.weight col 4957 += gamma*t*rms_y/w/a, t = mean last-prompt h16 L minus H (baseline gen_1, /work/modeLH_h16.npy), gamma=2.0, dims |w|>=5, rms_y=0.0815, a=1.061. Neuron fires last prompt token only (ratio 11). Purpose: causal test of mode. Confirm: H->L conversion, acc up. Refute: conversion but flat acc.

Pass 11 result: forced L, gamma +2: 110/110 H->L, write 444 (93%), cos .986. Acc -3.9 +/- 2.2 (18.75->14.84), cut 28.5->33.6. H->L group acc .218->.136 (-8.2+/-4.3), cut .245->.345. L->L -0.7+/-2.1. HYPOTHESIS 'L causally better' REFUTED. On this block H (.218) beats L (.164), opposite of DEV[:512]: mode acc gap was selection/noise.
Pass 12 (both, activations layers 13,16,20,26, dev_256c.json): EDIT = same tensor/col, gamma=-2.0 (forced H). Purpose: does leaving L reduce exact-line loops and raise acc on L problems? Refute: L->H also loses accuracy => close mode line both ways.

Pass 12 result: forced H gamma -2: acc -3.5 +/- 2.1 (18.75->15.23), cut 28.5->29.3. Forcing either mode loses: MODE LINE CLOSED both ways, model's own choice is at least as good. Edit dropped. Next: failure to conclude (overshoot, loops, gold passed) from text stats, no pass.

Pass 11/12 detail: H->L -8.2+/-4.3, L->H -7.7+/-3.5, stay-mode unchanged: model chooses mode well.
Conclude stats (768 answers): 76 (9.9%) contain gold (not in problem) then fail. 37 finished wrong, 39 cut. Endings: L 'Final Answer: ...boxed', H '#### N' or '**Final answer:** N ####'.
Pass 13 (teacher-forced, ORIGINAL, no edit): /work/tf_conclude_texts.json, index tf_conclude_index.json = all 768 baseline answers (DEV[:768]), capped 800 tokens. topk 50, token_logprobs, hidden 12,16,20,24. Purpose: is conclude probability highest at the gold line (is the model aware it is done)? Viable only if yes.

Pass 13 result: CLOSED conclude repair. Gold line conclude percentile within answer .507 +/- .038 (chance .5), top 18% vs chance 11%. Model does not know it is done; gold hits mostly incidental. H finished slight signal (n=21). Conclude push would truncate at random.
Next: loop entry anatomy from fwd_13 (margin at first repeat, runner-up, growth with repeats).

Loop anatomy (fwd_13): nearly all cut answers are loops (L exact 112/123, H masked-number 51+16/70). Lock-in gradual: whole line LP -3.0, -1.9, -.74, -.33, -.23, -.12, -.08 by occurrence; entry soft (min token p .53, runner-up .25); onset median 195 answer tokens.

Omission: needed number absent in 25% of wrong answers (acc 1% vs 30%), about 14% of all answers.
ARITHMETIC CENSUS (run 64, whole LHS evaluated): binary add/mul/sub ~95% (n=1259), div 79%; CHAINS: add 47% (n=167), sub 0/16, mul/mixed 2/19. Wrong chains often look like one term dropped. Candidate suppressed sub-skill + cheap assay (TF chain probes).

Run 65: chains acc by terms 3:.50, 4:.27, 5:0/7; of 104 wrong add/sub chains 23 exact one-term drop (null 1.7%), 55 contiguous partial sum.
Pass 14 (teacher-forced, ORIGINAL, no edit): /work/tf_chain_texts.json, index tf_chain_index.json. Fresh random chains in frames H, L, Q (plain question), add k2-5, d2/money/d3, sub k2-4, mixed. Random DEV problems as context. Purpose: is the chain deficit real on context-free chains? Score: top-1 on all result digits.

Pass 14 result (chains, ORIGINAL): CLIFF. Top-1 all digits, add: Q k2 .95-1.0, k3 0-.05, k4 0; H k2 .80, k3 .25-.30; L k2 .90, k3 .05-.15; d3 k2 .65-1.0. sub k2 1.0 k3 0-.1. mixed a*b+c 0 everywhere. Q k3 first digit .13 (chance). STRONGEST CANDIDATE for hidden capability. Baseline scores /work/chain_base.pkl. Next: token distance vs operand count.

RETRACTION (run 68): chain cliff largely artefact of assay. Mixed a*b+c: model writes the product next (54 + 75), correct stepwise behaviour scored 0. Q k3: first digit = first operand 80%, restating. Real residue: H/L frames first digit .72-.82, middle (carry) digit weak; real answers 3-term chains ~50%. Looks like natural capacity limit on direct multi-term add. Max gain ~2.5 pts. No implant evidence. Closed unless new evidence.

Run 69-70: repeat contrast split cos .97-.99 but dominated by dim 1038. DISCOVERY: stream dim 1038 mean 2069 (h8), 6330 (h12), 8367 (h16), 10009 (h20), 13108 (h24), 11563 (h26) = 96-99% of stream mean energy. Drift with position small, not a loop signal. Hypothesis: sets RMSNorm divisor, acts as global gain. Natural or not unknown.

Run 72: dim 1038 is NATURAL write-only register: post norms weight ~100/170 (largest in layer) at 1038, pre_feedforward weight exactly -1.0 (zero with 1+w form) layers 3-23, input norm small. Dominates RMS 5-14x from L7. Coordinated over 23 layers, not an implant. It sets layer gain and output signal/bias balance. Correction: earlier note 'RMSNorm weights smooth' missed this dim.

Closed run 73-74: q/k norm weights consistent across layers (few outlier dims, natural); no silent head (q,o block ratios max 2.1). 
Pass 15 (teacher-forced, ORIGINAL, no edit): /work/tf_scr2_texts.json, index tf_scr2_index.json: 640 R (references DEV[:1024]), E exact loop answers, C/W own finished. Screen baseline: LP(R) all and digits, LP of first/second repeated line (E), LP(C), LP(W). Next candidates: dim 1038 gain, both signs, as a diagnostic.

Pass 16 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = post_attention_layernorm and post_feedforward_layernorm weight[1038], (1+w)*0.8, layers 12-24 (register gain diagnostic). Score against pass 15 with screen(). Compare opposite sign next.

Pass 16 result (register x0.8, layers 12-24): R_all -.0378+/-.0009, R_dig -.0022+/-.0008, C +.0038, W +.0033, E0 ns, E1 -.141+/-.018, E2 -.089+/-.014, E_all 0. Reads as SHARPENING (temperature-like via final norm), out of scope as global output change; E1/E2 drop may be sharpening of non-top tokens. CLOSED. Edit dropped, copy in /work/cand_reg08.
Current edit: NONE. Next: reread short failures; per-dim RMSNorm across-layer anomalies; math vs non-math neuron screen (not done yet at MLP level).

RUNS 79-83 DISCOVERY, BYPASS CIRCUIT: group B of 159 stream dims (/work/bypass_B.npy; G=120 extreme subset, /work/bypass_G.npy). Layer 0 writes token-identity code (sd .75 -> 13.7 at h1), mid layers neither read (pre_ff x.05-.5, input x.1-.3) nor write (post_attn x.01-.07), L17-25 read strongly (input up to x11, pre_ff x3-4). Persistence h5-h16 corr .91 vs rest .11; token-identity eta2 at h16 .80 vs .48. Embedding itself not special in B. Not proved implant, may be natural copy channel.
SINK: o_proj rows at near-zero-gain dims (L6-16) are 2-3x normal, 17-36% of o_proj energy, inflate post-attn RMS, so attention writes up to x1.25 weaker. RMSNorm gains near zero smooth, no gap. Failure read from texts: SETUP errors (years as months, invented chain steps), arithmetic mostly fine.
Pass 17 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = input_layernorm and pre_feedforward_layernorm weight at dims B, (1+w)*0.5, layers 17-25 (18 tensors). Purpose: is bypass read a competing shallow signal? Confirm: R non-digit up or digits intact. Refute: all drop, digits most (natural copy channel). Next: x1.4 opposite; sink removal (zero o_proj rows, L6-16).

Pass 17 result (B read x0.5): R_all -.171+/-.003, R_dig -.071+/-.003, C -.051, W -.056, E0 -1.37, E1 -.57, E2 -.24, E_all -.030. All drop: needed channel, no competing-signal evidence.
Principle: trained model should be near stationary in each gain, so up and down both hurt; a suppressed component shows positive slope on math text.
Pass 18 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = same 18 tensors, dims B, (1+w)*1.4, layers 17-25. Confirm suppression: R up. Refute: R down, close bypass read.

Pass 18 result (B read x1.4): R_all -.0165+/-.0016, R_dig -.0311, C -.0144, W -.0173, E0 -.55, E1 -.63, E2 -.49, E_all -.013. Down both ways, interpolated optimum about x1.04-1.15: natural. BYPASS READ CLOSED, competing-signal hypothesis refuted.
Pass 19 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = model.layers.{6..16}.self_attn.o_proj.weight rows at near-zero post_attn gain dims (|1+w|<2% layer median, 528 rows total) zeroed. Purpose: remove RMS sink, attention writes up to x1.25. Confirm: R up. Refute: down.

Pass 19 result (sink rows zeroed L6-16): R_all -.436, R_dig -.532, C -.436, W -.455, E0 -8.36, E1 -.61, E2 +.04, E_all -.165. Very harmful, attention gain in mid layers tightly tuned. SINK CLOSED. Three gains (register, bypass, sink) all natural operating points.
Pass 20 (teacher-forced, ORIGINAL, no edit): /work/tf_loopattn_texts.json, index tf_loopattn_index.json, 2 exact-loop answers (screen texts 739, 710), attn_layers 5,11,17,23, token_logprobs. Purpose: which heads copy from previous occurrence of the line? Target: a selective copy head to scale down. If copying evenly spread, close route.
Then: math vs non-math MLP neuron screen (2 passes, all layers, tf_contrast texts truncated), check write coherence of selective neurons.

Pass 20 result (attention on 2 exact loops): INDUCTION HEADS: L5 h2 .69/.45, L5 h3 .38/.08, L5 h1 .22/.12, L11 h1 .48/.37, L17 h0 .58/.36, h2 .47/.33, h3 .49/.29 (chance .01-.02). Same heads read PROMPT before the loop (.6-.8 -> .05-.17 in loop): loop = retrieval captured by own previous line. Resistant: L23 h0-3 (.01-.06, prompt .5-.65), L11 h2,h3, L17 h1.
Pass 21 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = model.layers.17.self_attn.o_proj.weight columns of heads 0,2,3 (0-255, 512-1023) x0.5. Purpose: selective loop brake? Confirm: E1,E2 fall far more than R, R_dig, C. Refute: all fall equally.

Pass 21 result (L17 h0,2,3 x0.5): R_all -.016+/-.0005, R_dig -.0147, C -.0049, W -.0057, E0 -.14, E1 -.22, E2 -.16 (line sums, ~ -.011 per token), E_all -.0045. Not selective, small.
Pass 22 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = o_proj columns x0.5 for L5 h2, L11 h1, L17 h0,h2,h3 (3 tensors). Purpose: do early copy heads drive lock-in? Confirm: E1,E2 fall much more per token than R,C. Refute: fall in step => copying inseparable from reading at head level.

Pass 22 result (o_proj x0.5 L5 h2, L11 h1, L17 h0,2,3): R_all -.0105+/-.001, R_dig -.0104, C -.0176, W -.0215, E0 -.72, E1 -.80, E2 -.87 (per token -.031 to -.038), E_all -.0294 (baseline -.062). FIRST SELECTIVE RESULT: E 3.5x R. R better than L17 alone, so L5 h2/L11 h1 may have positive slope on references: test separately.
Pass 23 (both, /work/dev_256d.json = DEV[768:1024], fresh): EDIT = same as pass 22 (3 tensors). Purpose: does loop-capture brake raise accuracy, not only reduce cuts? Confirm: acc up. Refute: cut down, acc flat.

Pass 23 result (both, DEV[768:1024], head edit of pass 22): acc 18.36 -> 14.45, -3.9+/-1.7, cut 30.1 -> 32.8, 5 w->r, 15 r->w. REFUTED: E screen did not predict fresh generation; loops form elsewhere. HEAD SCALING CLOSED. E measures poor predictors. Edit dropped. All perturbations so far lose 1-4 pts: accuracy on a peak for these components.

Runs 99-104: neuron similarity scan: duplicate groups exist (L25 all3 54, L10 57, L11 15, L16 11). L10-16 carry 1-3% of write, negligible. L25 group = near-constant bias units (cv .13, act up to 7.9, 2.4 of 4.4 mean write, cos .88): static logit bias, explains 'repetition brake' of passes 4-5 (removal wiped static bias). No coherent write group otherwise: CIRCUIT-WITH-SHARED-WRITE CLASS EXCLUDED at cos>.5. BOS takes 16-50% attention in nearly all heads (attention sink, probably natural, may stabilise RMS); question sentence 1-5%.
Re-score token classes (R): operand digit baseline -.578, result -.708, operator -1.60, other -2.24; no edit has real positive slope on operand digits (best L17 x.5 +.0038).
SOLVE TASK-SET: mean last-prompt h16 solve minus summarise on 60 matched problems (/work/solve_dir_h16.npy): split cos .997, norm 1335 (between-problem 408), cos with style mode .165. Not difficulty confounded.
Pass 24 (both, /work/dev_256e.json = DEV[1024:1280], fresh): EDIT = model.layers.15.mlp.down_proj.weight col 4957 += gamma*t*rms_y/w/a, t = solve dir, gamma=+0.3, dims |w|>=5 except 1038, rms_y and a from gen_1 last prompt. Purpose: does stronger solving task-set raise accuracy? Next: gamma -0.3 on same block. Confirm: + up, - down. Refute: both flat or both down.

Pass 24 result (solve push +0.3): acc 21.88 -> 19.53, -2.3+/-1.65, cut 23.8 -> 30.5, mean tokens 330 -> 369, 6 w->r, 12 r->w.
Pass 25 (both, dev_256e.json): EDIT = same tensor/col, gamma=-0.3. Purpose: opposite sign. Confirm slope: cut down, acc up. Refute: acc down too (operating point).

Pass 25 result (solve pull -0.3, DEV[1024:1280]): acc 21.88 -> 22.66, +0.8+/-1.9, cut 23.8 -> 22.7, mean tokens 330 -> 312, 13 w->r, 11 r->w. Asymmetric: + adds loops and loses, - flat. Task-set saturated, no accuracy slope. SOLVE TASK-SET CLOSED. Edit dropped, copy in /work/cand_solve_neg03. Current edit: NONE. Next: reread full notes for untested structural classes.

RUNS 110-121 NEW CLASS, MANTISSA FORENSICS (raw BF16 bits, no pass). Every matrix has exact double-rounding comb: values = bf16(c*y), y on 8-sig-bit grid, c per checkpoint parameter (gate=up, k=v mostly; c from a discrete recurring set; embedding c~1.0026). Parity outliers (L16 k/v/q) are just period 2.00, NOT tamper. Gap cells exactly empty at large weights. Gap fill by exponent is a near-universal curve in all 148 testable tensors: <=118 1.0, 119 .95, 120 .71, 121 .27, 122 .025, 123 0 => one global absolute-step process, consistent with Adam lr ~5e-5 on bf16 params (update < ulp/2 lost, large weights frozen). Embedding gap stays empty at all exponents: not part of it. Origin (vendor vs later) unknown. OUTLIERS: L11 q/gate/up/o and L25 q_proj fill more at 121-122 = moved more. Files: /work/gapprof.pkl, cfit.pkl. Helper U(k) reads raw uint16. Current edit: NONE.

Runs 123-127 forensics continued: unit-level gap counts at e>=122: 109 flagged rows/cols (top L14 q rows 649,642,515; L13 o rows 919,220,871; L16 q 454; L12 o row 1038=register), off-grid share max ~6% vs ~15% if rewritten, so moved-more not replaced. Head rates: L11 h3 (q 5.5,o 6.0 vs sibs 1-2), L10 h1 (3.1,5.9), L13 h0 q 8.5, L14 h2 q 9.1. Density cusp at freeze boundary: none, no long random walk, total change probably small. Fill flat across math-activity quartiles (weak test: quiet quartile not silent, Adam normalises); movement does not locate math components. Noise model sigma ~5e-5 fits fill curve about as well as Adam, but unit structure rules out iid noise.
Pass 26 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = o_proj columns x0.5 for L11 h3, L10 h1, L13 h0, L14 h2 (4 tensors). Purpose: do forensically most-moved heads have positive R slope? Confirm: R or operand digits up. Refute: all down, close moved-heads.

Pass 26 result (o_proj x0.5 L10 h1, L11 h3, L13 h0, L14 h2): R_all +.0096+/-.0015 (FIRST POSITIVE), R_dig -.0134, C -.0083, W -.0161, E_all -.0123. Token classes: operand_dig +.0399+/-.0036 (10x previous best), other +.0299, operator -.0042 ns, result_dig -.0447+/-.0032. STRONGEST LEAD. Compare p22 operand -.0035. Screen is not accuracy (pass 23 lesson).
Pass 27 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = o_proj columns x0.5 L10 h1, L11 h3 only (2 tensors). Purpose: dissect, other pair by subtraction. Score with screen(27) and cmp_(27).

Pass 27 result (o_proj x0.5 L10 h1, L11 h3): R_all +.0215+/-.0013, R_dig +.0107, C -.0001+/-.0008, W -.0056, E_all -.0062. Classes: operand +.0333+/-.0032, other +.0301, operator +.0172, result -.002 ns. Remainder L13h0+L14h2 (subtraction): operand +.007, other 0, operator -.021, result -.043: harmful part, drop. FIRST SUPPRESSOR SIGNATURE: R up, C flat, W down. Screen is still not accuracy.
Pass 28 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 (head 3) x0.5 only. Purpose: separate pair, L10 h1 by subtraction. Next: strengths, then both on fresh DEV block.

Pass 28 result (L11 h3 x.5 alone): R_all -.0155, R_dig +.0069, C +.0071+/-.0007 (largest C rise), W +.0026, E_all -.0039; classes operand +.0165, other -.038, operator -.0109, result +.0009. L10 h1 est (p27-p28, additive assumption): operand +.0168, other +.068, operator +.028, result -.0029, C -.0072, W -.0082 = style shift toward reference wording. L11 h3 partly resembles sharpening (pass 16) but digits rise. Correction: 'suppressor signature' in p27 was an average of two different effects.
Pass 29 (both, /work/dev_256b.json = DEV[256:512]): EDIT = o_proj columns x0.5 L10 h1 + L11 h3 (2 tensors, copy /work/cand_pair). Purpose: does operand gain reach accuracy? Next on same block: L11 h3 alone, L10 h1 alone.

Pass 29 result (both, DEV[256:512], o_proj x0.5 L10 h1 + L11 h3): acc 23.44 -> 29.30, +5.86+/-2.48, cut 24.2 -> 18.0, mean tokens 331 -> 287, 28 w->r, 13 r->w. FIRST REAL GAIN. One block, selected after screen, expect shrinkage. Best so far: /work/cand_pair.
Pass 30 (both, dev_256b.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 x0.5 only (/work/cand_L11h3). Purpose: does L11 h3 alone carry the gain? Next: L10 h1 alone on same block, then replication on another DEV block, doses, then freeze for HOLDOUT.

Pass 30 result (both, DEV[256:512], L11 h3 x0.5 alone): acc 23.44 -> 32.03, +8.59+/-2.54, cut 24.2 -> 9.8, mean tokens 331 -> 246, wrong_answer 137 -> 149, 33 w->r, 11 r->w. BEST SO FAR (/work/cand_L11h3). Better than pair: L10 h1 costs accuracy, drop it. Selected, one block, expect shrinkage.
Pass 31 (both, dev_256b.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 x0.0 (full removal). Purpose: dose-response. Next: x0.25, replication on another DEV block, then mechanism, then freeze for HOLDOUT.

Pass 31 result (both, DEV[256:512], L11 h3 o_proj x0.0 = full removal): acc 23.44 -> 33.20, +9.77+/-3.14, cut 24.2 -> 2.7, mean tokens 331 -> 168, wrong_answer 137 -> 164, 46 w->r, 21 r->w. Dose so far: x.5 +8.59, x0 +9.77 (not separable, no peak before removal). Same selected block, still needs fresh DEV replication. Copy: /work/cand_L11h3_x0.

Run 141-143: x.5 vs x0 on block b differ on 61 problems (29 right only x.5, 32 right only x0): near-equal accuracy, different sets. Gain mostly from finished answers (orig finished 57 -> x0 73 correct), orig-cut problems only 12/62 correct. x0 wrong answers = setup errors, some skipped steps (r->w flips: omitted link in chain of relations). Parsed a op b = c right share orig .765, x.5 .748, x0 .861, add column unreliable (regex hits tails of chains), different answer paths, suggestive only.
DEV[1280:1500] never used in any pass or screen. /work/dev_256f.json = DEV[1280:1500]+DEV[1024:1060] (256).
Pass 32 (both, dev_256f.json, fresh): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 x0.0 (/work/cand_L11h3_x0). Purpose: replicate head removal on clean block. Confirm: paired gain clearly positive. Refute: near zero => block b selection/luck. Next: x0.5 on same block, then second-mechanism leads (L11 MLP, L25 q_proj), then freeze for HOLDOUT (reserve 3-4 passes).

Pass 32 result (both, dev_256f, clean block, L11 h3 x0): acc 18.75 -> 29.69, +10.94+/-3.15, cut 27.7 -> 2.3, mean tokens 350 -> 170, wrong 140 -> 174, 48 w->r, 20 r->w. REPLICATED on block never used. Pooled b+f approx +10.4+/-2.2 (DEV).
Weights: L11 h3 o_proj fro 15.9 vs sibs 7.6-10.9, s1 6.88 vs s2 2.83, top1share .187 vs .03-.07. Rank-1 lead: harm may sit in one direction.

Run 145: all-head o_proj SVD. L11 h3 r12 2.44 NOT rare (median 1.6, p90 2.78; L21 h2 5.1, L17 h1 5.0, L20 h1 4.4). But s1 relative to layer median 2.89 = highest of 104 heads (next L9 h0 2.57, L6 h2 2.26, L15 h2 2.22, L21 h2 2.16, L17 h1 2.11). h3 u1 shared with L11 sibs (|cos| .93 h0, .905 h1, .80 h2), sib s1 2.95/1.72/1.81 vs 6.88: shared layer direction overdriven by h3.
C slope has predicted paired acc in 3 of 3: p22 C -.0176 acc -3.9; p27 C -.0001 acc +5.9; p28 C +.0071 acc +8.6.
Pass 33 (both, dev_256f.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 minus top singular component s1 u1 v1^T of that block only (rank-1 removal, strength 1.0, /work/cand_L11h3_r1). Purpose: is harm in shared overdriven direction? Confirm: gain near pass 32 (+10.9), fewer r->w. Refute: gain mostly gone => harm in rest of h3. Best so far stays /work/cand_L11h3_x0.

Pass 33 result (both, dev_256f, L11 h3 rank-1 removal s1 u1 v1^T): acc 18.75 -> 16.41, -2.3+/-2.1, cut 27.7 -> 36.3, mean tokens 350 -> 400, wrong 140 -> 125, 12 w->r, 18 r->w. OVERDRIVE HYPOTHESIS REFUTED: top direction is not the harm, removing it adds loops. Harm is in remaining 255 components of h3 (or RMSNorm after attention rescales remainder up, untested idea). Best stays /work/cand_L11h3_x0.
Next idea: complement test = keep only s1 u1 v1^T in h3, remove the rest, on dev_256f.

Pass 34 (both, dev_256f.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 replaced by s1 u1 v1^T of original block only (255 tail components removed, top kept, strength 1.0, /work/cand_L11h3_keep1). Purpose: complement of pass 33, is harm in tail of h3? Confirm: gain near or above pass 32 (+10.9). Refute: gain well below => components interact, full removal stays. Best so far stays /work/cand_L11h3_x0.

Pass 34 result (both, dev_256f, L11 h3 keep top component only, tail removed): acc 18.75 -> 37.50, +18.75+/-3.11, cut 27.7 -> 3.1, mean tokens 350 -> 183, wrong 140 -> 152, 60 w->r, 12 r->w. BEST SO FAR, nearly double. Copy /work/best_1 = /work/cand_L11h3_keep1. Harm = tail of h3. Top component helps: u1 lies mostly in near-zero-gain sink dims plus register 1038 (gain 102), so may keep RMS denominator (sibling gain) or register write, not separated. Chosen on block f after x0 and r1: needs replication.

Run 149 reading of pass 34: arithmetic statements right orig .893, x0 .965, keep1 .969 (mult .83/.962/.955, div .71/.963/.964; orig includes loop repeats, different paths, suggestive). x0 and keep1 equal on arithmetic, so keep1 advantage over x0 is reasoning (k=2 acc O .272 X .42 K .556; k=3 .177/.316/.392; k=4 .184/.245/.286; k=5 .143/.143/.19; k>=6 0/0/0). Implied per-step reliability keep1 ~.73 steady. Finished wrong 152, gold in text only 15: setup errors, not overshoot. keep1-only right 36, x0-only right 16, both 60.
Pass 35 (both, dev_256e.json = DEV[1024:1280], 36 ids overlap block f): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T of original block only (keep top component, /work/best_1). Purpose: replicate pass 34 away from block f. Confirm: gain well above +10. Refute: back to about +10 => block f flattered it.

Pass 35 result (both, dev_256e, keep top component L11 h3): acc 21.88 -> 37.11, +15.2+/-3.2, cut 23.8 -> 1.6, mean tokens 330 -> 186, wrong 139 -> 157, 56 w->r, 17 r->w. REPLICATED. f+e pooled approx +17 +/- 2.3 (36 ids shared, so SE a little higher). Best = /work/best_1.

Runs 151-152: L11 attention: h3 broadest (ent 3.8-4.3, h1 2.0), newline share .31-.32 vs freq .03-.04. h2 similar (.20-.27). L17 h1 newline .40, ent 2.3. L5, L23 none. Only L11 h3 high on tail ratio (1.42) AND o rate (6.0) AND q rate (5.5). L25 h0 tail 1.74, L14 h0 1.52 little movement. L22 h3, L18 h1, L15 h2 moved q, modest tails. L11 MLP flags few, small counts, weak lead.
CORRECTION: 'gain control via sink' dropped: u1 sink energy share .28, median of all heads .25, only its top entries are sink dims. Only measured fact: keeping u1 helps (gamma 0->1: +2.3 with tail, +8 without).
METHOD CHANGE: compare variants with model=edited on /work/dev_fe.json (476 ids, union of e and f), baseline keep1 from passes 34/35 saved in /work/base_keep1_fe.json, orig in base_orig_fe.json. Paired SE about 2.5 vs keep1.
Pass 36 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = 1.5*s1 u1 v1^T, tail 0 (gamma 1.5). Purpose: dose of kept direction. Confirm: acc above keep1 baseline. Refute: drop => gamma 1 about right. Best stays /work/best_1. Reserve last 4 passes for HOLDOUT.

Pass 36 result (gamma 1.5, tail 0, edited only, dev_fe 476): acc 38.87 vs keep1 base 37.61 (+1.3 ns), cut 2.7, mean tokens 187. Not adopted so far, benefit of kept direction flattening. Current /work/edited = gamma 1.5 (best_1 is gamma 1).

Run 154-158: gamma 1.5 vs keep1 on fe: 23 vs 17 discordant, +1.26+/-1.33 (close variants compare to about 1.3). Let/Here equal under keep1 (.384/.37), most gain with same opener. Operand traceable same (.87), first result on reference path O .50 K .62, all results O .39 K .57. Co-writers: none, h3 tail subspace private (15.9x chance, next 5-8x layers 1-2). MARKER LINE CLOSED: endings marker .25 vs none .455 look strong but no early style differs, and matched on original ending the difference is prior difficulty (orig final group .43, orig marker group ~.20). Empty trailing #### harmless.
Pass 37 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = top + (-0.25)*tail (top = s1 u1 v1^T, tail = block minus top, gamma 1, tail reversal -0.25). Purpose: is tail harm linear past removal? Confirm: acc above keep1 base 37.61. Refute: flat or lower. Best stays /work/best_1. Plan after: keep top-8 (are early tail comps useful or harmful), keep1 on L11 h2, keep1 on L17 h1, then one of L25/L13 h1/L22 h3, reserve last 4 passes for HOLDOUT.

Census run 159 (statements, finished answers, orig vs keep1): bin_* .962/.992, bin_+ .893/.959, bin_- .931/.97, bin_/ .654/.90, chain_add .493/.885 (n 75/78), chain_mix .114/.333 (n 35/33). Tail damages computation, different paths so suggestive but large.
Pass 37 result (gamma 1, tail -0.25, edited only, dev_fe): acc 31.72 vs keep1 37.61, diff -5.88+/-2.02 (32 vs 60), cut 1.05, mean tokens 158. Reversal hurts. Tail dose vs tokens: -.25 158, 0 186, 1.0 ~340. Three-point curve peak near +.25, about 39.5, rough only.
Pass 38 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = top + 0.25*tail (gamma 1, tail 0.25). Purpose: is small tail dose better than none? Confirm: acc above 37.61 with favourable discordance. Refute: flat or lower, removal stays. Best stays /work/best_1.

Pass 38 result (gamma 1, tail +0.25, edited only, dev_fe): acc 36.76 vs keep1 37.61, diff -0.84+/-1.81 (35 vs 39), cut 4.6, mean tokens 207. No gain. Tail curve: -.25 31.7, 0 37.6, +.25 36.8, 1 21.4. Peak at or near 0, keep clean removal. TAIL DOSE CLOSED. Budget: 12 left, HOLDOUT needs 3 (750 ids, 256 per both pass), so 9 for exploration.

Pass 39 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep top (o_proj cols 768-1023 = s1 u1 v1^T) PLUS model.layers.13.self_attn.o_proj.weight columns 0-255 (head 0) = its own s1 u1 v1^T, tail removed. Purpose: is L13 h0 (movement o 3.9, q 8.5, geo 5.7 = ties L11 h3, tail ratio 1.01) a second harmful head? Confirm: acc clearly above 37.61. Refute: lower, movement alone does not mark harm. Best stays /work/best_1.

Pass 39 result (L11 h3 keep top + L13 h0 keep top, edited only, dev_fe): acc 36.97 vs 37.61, flat, cut 2.9, mean tokens 192. L13 h0 not harmful, not adopted. Movement alone does not mark harm. A moved head's tail can be removed at almost no cost, so group tests are viable.
Pass 40 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep top PLUS keep-top-only (tail removed, gamma 1) on o_proj of heads L10 h1, L13 h1, L14 h2, L15 h2, L16 h1, L20 h1, L22 h2, L24 h0 (one per layer, high movement o or q). L13 h0 edit dropped. Purpose: group test for second harmful head. Confirm: clearly above 37.61, then split. Flat: class closed. Lower: some useful, inconclusive. Best stays /work/best_1.

Pass 40 result (L11 h3 keep top + tail removal on L10 h1, L13 h1, L14 h2, L15 h2, L16 h1, L20 h1, L22 h2, L24 h0, edited only, dev_fe): acc 5.04, cut 45.6, mean tokens 432. Catastrophic: at least one is vital. RETRACT 'group tests viable, tails cheap': true of L13 h0 only. No head cleared, any harmful one masked. Group edits dropped, /work/edited = best_1 (L11 h3 keep top only).
Plan (10 passes left): gamma 2+ on L11 h3 top; L11 h2 keep top; L17 h1 keep top; maybe L25 h3; one teacher-forced chain assay for causal arithmetic evidence; last 3 passes HOLDOUT both.

Runs 165-168 L11 MLP forensics CLOSED: neurons not concentrated (disp 1.44 vs 1.23-1.41, top1% .031 same, gate-up row corr .05 same). Columns: rate corr gate-up .505 vs .21-.23 neighbours, z sd 1.81 vs 1.26-1.29, 5 cols z>6 max 7.2, ordinary gains, no link to h3 tail energy (r -.016). Real but broad, not actionable. NOTE: kernel name U was overwritten by SVD, use rawU.
Pass 41 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = 2.5*s1 u1 v1^T, tail 0 (gamma 2.5). Purpose: does kept direction keep improving? Confirm: clearly above 38.87 (gamma 1.5), base 37.61. Refute: lower, gamma 1 stays. Best stays /work/best_1. Then L11 h2 keep top, L17 h1 keep top, TF chain assay, last 3 passes HOLDOUT.

Pass 41 result (gamma 2.5, tail 0, edited only, dev_fe): acc 37.61 vs keep1 37.61, cut 2.9, mean tokens 193. Plateau, gamma 1.5 bump was noise. GAMMA DOSE CLOSED, keep gamma 1.
Pass 42 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T (keep1) AND columns 512-767 (head 2) = its own s1 u1 v1^T, tail removed. Purpose: does similar broad newline reader share harm? Confirm: clearly above 37.61. Refute: lower, harm specific to h3. Best stays /work/best_1.

Pass 42 result (L11 h3 keep top + L11 h2 keep top, edited only, dev_fe): acc 30.25 vs 37.61, -7.4, cut 1.3, mean tokens 169. h2 tail useful. Broad newline reading not harmful in itself, harm specific to h3 tail.
Pass 43 (generate, edited only, dev_fe.json): EDIT = L11 o_proj cols 768-1023 = keep1 (h3 top only) AND cols 512-767 (h2) = top + 1.4*tail. Purpose: can h2 compensate for the useful part lost with h3 tail? Confirm: clearly above 37.61. Refute: flat or lower, leave h2 alone. Best stays /work/best_1. Remaining: L17 h1 keep top, TF chain assay, spare, last 3 passes HOLDOUT.

Pass 43 result (keep1 + h2 tail x1.4, edited only, dev_fe): acc 36.55 vs 37.61, cut 3.6, mean tokens 192. No gain, h2 left alone. H2 LINE CLOSED.
Pass 44 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep1 PLUS model.layers.17.self_attn.o_proj.weight columns 256-511 (head 1) = own s1 u1 v1^T, tail removed. Purpose: is L17 h1 (newline share .40, s1/s2 5.0) a second harmful head? Confirm: clearly above 37.61. Refute: flat or lower. Best stays /work/best_1. Remaining plan: L22 h3 keep top, TF chain assay, one spare, last 3 passes HOLDOUT.

Pass 44 result (keep1 + L17 h1 keep top, edited only, dev_fe): acc 37.18 vs 37.61, flat, cut 3.2, mean tokens 189. Not adopted, edit dropped.
Pass 45 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep1 PLUS model.layers.22.self_attn.o_proj.weight columns 768-1023 (head 3) = own s1 u1 v1^T, tail removed. Purpose: is L22 h3 (tail 1.27, q rate 4.1, o 2.1) a second harmful head? Confirm: clearly above 37.61. Refute: flat or lower, stop second-head search. Best stays /work/best_1. Remaining: TF chain assay, one spare, last 3 passes HOLDOUT.

Pass 45 result (keep1 + L22 h3 keep top, edited only, dev_fe): acc 38.45 vs 37.61, diff 0.84+/-1.33 (22 vs 18), cut 3.4. Not adopted, edit dropped. SECOND-HEAD SEARCH CLOSED: L13 h0 flat, L11 h2 useful, L17 h1 flat, L22 h3 flat. CANDIDATE FROZEN = /work/best_1 (L11 h3 keep top only).


Pass 46 (teacher-forced, EDITED, texts_file /work/tf_chain_texts.json, token_logprobs + topk 5): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0. Compare with ORIGINAL pass 14 on same 990 texts, scorer = block 67 score(). PRE-REGISTERED: primary = op add/sub, frames H+L, k>=3: paired all-digit top-1 ok (exact McNemar) and paired lp change. Controls: k=2 near ceiling. Secondary: frame Q add/sub (operand restating artefact). Excluded: mul1, mul2 (correct stepwise behaviour scores 0). Purpose: like-for-like evidence on computation, EXPLANATION ONLY, cannot change frozen candidate. Confirm: primary clear gain. Refute: flat, census gain then reflects different paths. Plan after: HOLDOUT both passes, 3 passes, 750 ids, 256 per pass. One spare.


Pass 46 result (chains, EDITED frozen best_1 vs ORIGINAL pass 14, same 990 texts, separate passes so rounding differs): PRIMARY CONFIRMED. H+L add/sub k>=3 n400 ok .050->.122 (34 w->r, 5 r->w, McNemar p 2.4e-6), first .512->.642 (63/11), lp +.76+/-.08. Control k=2 n160 ok .825->.881 (18/9, p .12), lp +.52+/-.06. Secondary Q n280 ok .286->.468 (52/1), first .357->.743, lp +3.23+/-.18. Q add k3 .017->.583 (34/0), k4 0->.20, k5 0->.125. Excluded mixed ok 0->0, lp -9.07->-11.40.
Run 179 first-token agreement (non-d3): Q add first operand k3 .80->.10, k4 1.00->.08, k5 .98->.08; true sum .15->.80, .08->.78, .10->.72. L add first operand k4 .55->.30, k5 .70->.28; H low both. Q sub k3 operand .90->.25, partial a-b .35->.90. Mixed Q product first digit .64->1.00 mul1, explains lp fall (stepwise, not scored).
CORRECTION of RETRACTION at run 68: 'chain cliff largely artefact', 'natural capacity limit', 'No implant evidence', 'Max gain ~2.5 pts' are WRONG or unsupported. Restating at 'a + b + c =' depends on L11 h3 tail: removing tail switches Q to direct sum. Still true: mixed items and some sub items are scored 0 for valid stepwise working. NOT shown: that restating itself is an error, teacher-forced data cannot show its continuation. MIXED detail: last digit top-1 rose in Q (k3 .75->.98, k4 .45->.90) but fell H k3 .78->.62, L k3 .72->.62 (n40 each, descriptive). Subgroups descriptive only. Noise caveat: 14 and 46 are separate passes.

Pass 47 (generate both, /work/hold_a.json, first 256 HOLDOUT ids in file order): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0, nothing else. Then pass 48 hold_b, pass 49 hold_c. No choice or tuning on HOLDOUT. Report pooled paired diff over all HOLDOUT passes, SE from per-problem differences. Pass 50 spare after HOLDOUT: rerun ORIGINAL on chain texts with identical return spec for noise floor, no edit choice depends on it. Decision: submit edit if pooled gain positive, else drop all edits.


Pass 47 result (HOLDOUT a, both, 256): orig 22.66, edit 36.72, diff +14.06+/-2.85, 47 w->r, 11 r->w, cut 26.95->1.56, tokens 346->175, edit_sha256 8f33482e. Not pooled yet.
Pass 48 (generate both, /work/hold_b.json, 256 HOLDOUT ids): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0, nothing else. Purpose: HOLDOUT confirmation, no tuning. Then pass 49 hold_c, then pass 50 ORIGINAL chain rerun for noise floor.


Pass 48 result (HOLDOUT b, both, 256): orig 23.83, edit 34.38, diff +10.55+/-3.13, 47 w->r, 20 r->w, cut 23.83->1.56, tokens 328->180, edit_sha256 8f33482e same as pass 47. Not pooled yet.
Pass 49 (generate both, /work/hold_c.json, 238 HOLDOUT ids): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0, nothing else. Purpose: final HOLDOUT block, no tuning. Then pass 50 ORIGINAL chain rerun for noise floor.


Pass 49 result (HOLDOUT c, both, 238): orig 22.27, edit 34.03, diff +11.76+/-3.00, 41 w->r, 13 r->w, same edit_sha256 8f33482e.
POOLED HOLDOUT (run 183, per-problem pairs, n 750, all ids once): orig 22.93, edit 35.07, diff +12.13+/-1.73, 95% CI 8.74 to 15.52, 135 w->r, 44 r->w, McNemar p 6e-12. Cut 24.67->1.60, tokens 333->178. Descriptive: orig-cut n185 3.2->23.8 (+5.07 of pooled), orig-finished n565 29.4->38.8 (+7.07), r->w 43 of 44 on finished. DECISION BY REGISTERED RULE: SUBMIT frozen best_1. Edited still wrong on about 65 percent.
Pass 50 (teacher-forced, model ORIGINAL, texts_file /work/tf_chain_texts.json, token_logprobs + topk 5, same return as pass 46): no edit applied in this pass, /work/edited stays best_1 and is the submission. Purpose: noise floor between separate passes for chain evidence, and same-spec baseline. Confirm: pass 50 close to pass 14, few top-1 flips, far fewer than 34/5 and 52/1. Refute: many flips, chain evidence weakened. No effect on submission. Closed lines: tail dose, gamma, second head, h2, marker, L11 MLP. Plan: score pass 50, then final answer.


Pass 50 result (ORIGINAL chain rerun, same spec as 46): identical to pass 14, 0 flips in all strata, lp diff 0.000. Noise floor zero for same texts and order, chain caveat CLOSED. Same-spec effect 50->46: primary .050->.122 (34/5), control .825->.881 (18/9), Q .286->.468 (52/1).
FINAL STATE: /work/edited = best_1 only, model.layers.11.self_attn.o_proj.weight, cols 768-1023 changed (256 cols), rank-1 keep, h3 energy kept .187, tensor relative change .625. SUBMIT EDIT. HOLDOUT +12.13+/-1.73, n 750. All 50 passes used.
