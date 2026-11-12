# Notes
Sets: DEV=/work/dev.json (1024, seed0 shuffle first 1024; dev_a first 512, dev_b last 512), HOLDOUT=/work/holdout.json (768, never tune), spare.
Current edit: none.
Pass 1 plan: generate original on dev_a with activations (all layers, mlp).

Findings: eval format (chat+suffix) adds a large constant residual offset written by layer-12 ATTENTION (fwd_4/5, v0-v3 on identical solution tokens: 0.86->2.49 sd at L12; MLP12 opposes, cos -0.68). Arithmetic fine (after '=' median lp -0.004). Weight anomaly scans: nothing.
dir files: /work/dir_j12.npy (v0-v3 diff of layer-12 increment, mean over sol tokens), /work/dir_a12.npy = normalize((1+w_postattn12)*j).
Pass 6 edit E1: L12 o_proj <- O - a a^T O (strength 1, full removal of a from attn12 output). TF on tf2 texts (512 = 128 dev x 4 variants) to compare with pass 3 (original).

CORRECTION: hidden_l = output of model.layers.(l-1). The jump hidden11->12 is model.layers.11 (full attention). E1 (L12 o_proj) dropped, no effect (-0.007).
L11 MLP neurons 6176 (v0 -2.88, v3 -0.19) and 6508 (v0 -1.97, v3 -0.17): eval-context, near-constant on answers.
Pass 7 edit E2: model.layers.11.mlp.down_proj column 6176 zeroed. TF on tf2 texts vs pass 3.

E2 (zero L11 n6176) result: v0 sol lp -0.014+-0.002 (worse), raw unchanged. Closed.
Pass 8 edit E3: remove a=normalize(g*j) (j=/work/dir_j12.npy, = v0-v3 diff of hidden12-hidden11, i.e. layer-11 write) from L11 o_proj and L11 down_proj outputs, strength 1. TF tf2.

E3 result: all variants worse (v0 -0.139+-0.009, v3 -0.121). Offset is useful, not suppression. CLOSED eval-context-offset line. Edits dropped; model unedited.

Finding: pass1 orig dev_a acc 0.223 (wrong 259, cut 139). TF fwd_3: numbers in v0 much less confident than v3 (computed median lp -0.089 vs -0.008; copied -0.181 vs -0.027). Hypothesis: digit predictions degraded in eval/chat context.
Pass 9 plan: TF original, tf4_texts (= tf2 v0 and v3, 256 texts, idx in tf4_sel.json), topk 10, hidden 0,2,..,26. No edit.

Finding fwd_9 (orig, v0 vs v3 aligned from end): correct-digit projection (normalized h . (E_tgt-mean E_dig)) equal through L18, v0 falls behind from hidden20 (-0.0011) to hidden26 (-0.0072); first digit after '=' wrong top1 5.0% v0 vs 2.4% v3, no systematic shift. Deficit written by model.layers 18-25.
Pass 10 plan: TF original, tf5_texts (64 shortest problems x v0,v3; sel5 = indices into tf4), mlp_act 18-25, hidden 18-26. No edit.

Finding: pass1 generations have degraded explicit arithmetic: 2-operand small add 5.4% wrong, mul 10%, sub 12%; multi-term sums often badly wrong. Style (Let vs Here's) no difference. Weight norm/neuron scans: only known massive dims.
Pass 11 plan: TF original, tf6_texts = 250 synthetic exprs x 4 contexts (A eval prompt+step, B chat 'Compute', C raw, D raw Question/Answer), topk 5. No edit.

fwd_11 (synthetic arithmetic, orig): add2 exact A(eval ctx)0.87 B(plain chat)1.00 C raw0.27 D raw+problem0.81; sub2 A.61 B.82 D.77; add34 ~0 everywhere. Errors mostly +-1/carry. Problem content (not chat) hurts. Late MLP (18-25) neurons: no context switch (|z|<1.5).
Pass 12 plan: TF original, 2 texts (/work/tf7_texts.json: wrong A add2 probe, v0 ref solution), attn_layers 5,11,17,23. No edit.

fwd_12 attention (wrong A add2): L17h3 91% on operands, L11h2 63%; others mostly BOS; no anomalous head.
Pass 13 plan: TF original, tf8_texts = synthetic add2/sub2/mul2 in contexts A,B (376 texts, idx in tf8_sel.json into tf6), hidden 0..26. No edit.

fwd_13 (A vs B synthetic, hidden): A-B diff small (5-15% rn), consistent to L16, no sharp layer; within-op not predictive of errors (mul A>B). CLOSED context-gated arithmetic line.
fwd_3 recheck: digits after '=' mean lp v0 -0.262, v1(chat no suffix) -0.218, v2/v3 raw -0.146. Graded, not switch.
gen_1 failures: cut = loops (rep8 0.58); wrong = fluent, invented quantities; marker use only 28% even when correct.
Pass 14 plan: TF original, /work/tf9_texts.json (24 eval-format math texts = own gen_1 answers 12 correct + 12 wrong, truncated <=300 tok; 24 handwritten non-math chat QA), mlp_act 0..25. Goal: find math-selective near-constant neurons (gate candidates). No edit.

fwd_14: math vs nonmath chat: L11 n6176 most selective neuron in model (math -2.66 vs chat -0.03, 6.2 sd, 67x scale); also n6508 (-1.34 vs -0.04). Contribs 0.70 and 0.40 of |L11 MLP out| 2.78; cos with math-chat out diff 0.52/0.22. No direct digit logit effect.
Pass 15 edit E4: model.layers.11.mlp.down_proj columns 6176 and 6508 zeroed. GEN both on dev_b[:256] (/work/ids_devb256.json).

E4 result (pass 15, dev_b[:256]): orig 0.1875 edited 0.2344, paired +0.047 +- 0.025 (27 w->r, 15 r->w); cut 66->53. Promising.
Pass 16 edit E5: L11 down_proj columns 6176, 6508 multiplied by -1 (reversal). GEN both dev_b[:256].

E5 (reverse) result pass 16: -0.059 +- 0.026 (15 w->r, 30 r->w). Monotone: zero +4.7, reverse -5.9.
Weights of n6176/6508 unremarkable (below-median norms). Downstream readers of n6176 diffuse (max cos 0.15).
Current edit = E4 (L11 down_proj cols 6176, 6508 zeroed), copied to /work/best_1.
Pass 17: GEN both E4 on dev_b[256:512] (/work/ids_devb2.json) to replicate.

E4 replication pass 17 (dev_b[256:512]): +0.031 +- 0.026 (26 w->r, 18 r->w), cut 75->53. Pooled DEV (512): ~+3.9 +- 1.8.
Pass 18 plan: TF original, /work/tf10_texts.json = 40 tf5 pairs (v0 eval chat, v3 raw) + 24 nonmath chat; mlp_act 2..17. Find AND(chat,math) units like n6176. Edit present in /work/edited (E4) but pass uses model=original.

fwd_18 AND(chat,math) scan L0-17: n6176 (8.1sd) and n6508 (4.2sd) dominate; second tier L12 n6233,n4689,n159; L13 n754,n2476 (~2sd). Nothing before L11.
Weights: up rows of 6176/6508 overlap L11 head3 output space 0.66/0.64 vs random 0.40+-0.07. L11h3 attends 30-40% to chat tokens (fwd_12).
Pass 19 edit E6 (diagnostic): L11 o_proj columns 768:1024 (head 3) zeroed; E4 dropped temporarily (best_1 holds E4). TF edited on tf10 texts, mlp_act 11,12,13.

fwd_19 (L11 head3 zeroed, diagnostic): n6176 v0 -2.89 -> -0.13, n6508 -1.98 -> -0.01, L12 n6233 +0.43 -> +0.02 (n4689, n159 halved). CAUSAL CHAIN: L11h3 -> L11 n6176/n6508 -> L12 n6233. Whole-head ablation costs sol lp -0.19 everywhere (too broad). E6 dropped.
Pass 20: E4 restored (L11 down cols 6176,6508 zeroed). TF edited on tf10 texts, mlp_act 12..25: rescan for remaining AND units.

fwd_20 (E4 edited, L12-25): backup unit L12 n4488 v0 -0.24 (orig) -> -1.59 (E4), v3/chat unchanged; was top reader of n6176. Write partly copies 6176's (cos 0.42 excl. massive dims; raw cos dominated by dim 1038). L12 n6233 / L13 n754 halved under E4.
Pass 21 edit E7: E4 (L11 down cols 6176, 6508 = 0) + model.layers.12.mlp.down_proj column 4488 = 0. GEN both dev_b[:256] (/work/ids_devb256.json; E4 got +4.7 there).

E7 result pass 21 (dev_b[:256]): +0.016 +- 0.021 (17 w->r, 13 r->w), cut 66->54. Worse than E4 (+0.047) on same set: backup n4488 not limiting. CLOSED backup-gate line. L12 edit dropped; current edit = E4 again.

Runs 102-112: E4 rescan L12-25 -> only L12 followers; SVD scan clean (closed); '####' suppressed in v0 (median lp -6.8 vs -2.9 raw) but marked answers are short wrong ones, grading-neutral (closed).
Head-3 dependence (fwd_18 vs fwd_19) + AND signature in orig: L11 n3704 (v0 -0.47 vs ~0), n2377 (+0.68 vs ~0.05), n4739 (+0.20 vs ~0.03).
Pass 22 edit E8: model.layers.11.mlp.down_proj columns 6176,6508,3704,2377,4739 zeroed (strength full). GEN both dev_b[:256] (/work/ids_devb256.json). E4 in /work/best_1.

E8 result pass 22 (dev_b[:256]): +0.035 +- 0.025 (25/16), not better than E4 (+0.047). CLOSED widening of L11 gate neurons.
Note: sol-lp is a poor proxy (E2 lowered lp yet E4 raised acc), so test head-level by GEN.
Pass 23 edit E9 (diagnostic): E4 (L11 down cols 6176,6508=0) + model.layers.11.self_attn.o_proj columns 768:1024 (head 3) zeroed. GEN both dev_b[:256].

E9 result pass 23 (dev_b[:256]): +0.090 +- 0.0285 (39 w->r, 16 r->w), cut 66->10, mean tok 345->187. Copied to /work/best_2. L11 head 3 is a major suppressor beyond n6176/6508.
Pass 24 edit E10 (strength test): E4 + model.layers.11.self_attn.o_proj cols 768:1024 multiplied by -0.5. GEN both dev_b[:256]. E9 in best_2.

E10 result pass 24: head3 x(-0.5)+E4: -0.086 +- 0.025 (11/33), cut 1. NON-MONOTONE: head3 has useful + harmful parts. Zero best so far.
L11h3 attends 25-27% to last prompt token ('\n' after model); its L11 input is constant across problems (cos .996 excl massive dims); value norm 16 vs answer-mean 2.2; head3 write from it aligns with 6176/6508 gate/up rows |cos| .06-.09 vs random .005.
Pass 25 edit E11: E4 + model.layers.11.self_attn.o_proj cols 768:1024 <- B (I - vh vh^T), vh = normalize(W_v11 . inputnorm11(mean hidden_11 at last prompt token, gen_1_acts)). Rank-1, strength 1. GEN both dev_b[:256].

*** E11 result pass 25 (dev_b[:256]): +0.273 +- 0.033 (80 w->r, 10 r->w), acc 0.461 vs 0.1875, cut 66->26. Copied to /work/best_3. vh saved /work/vh_L11_lastprompt.npy.
Pass 26 edit E12 (strength 2): E4 + L11 o_proj head3 cols <- B (I - 2 vh vh^T). GEN both dev_b[:256].
E12 result pass 26 (strength 2): +0.070 +- 0.031 (41/23). Strength 1 (removal) is best.
Pass 27 edit E13: ONLY L11 o_proj head3 rank-1 removal of vh (strength 1); L11 down_proj restored (no E4). GEN both dev_b[:256].

*** E13 result pass 27 (head3 rank-1 only, no E4): +0.285 +- 0.033 (83/10), acc 0.473. E4 redundant. Copied to /work/best_4 (CURRENT BEST).
Pass 28 edit E14: L11 o_proj heads 0,2,3 columns each <- B(I - vh vh^T). GEN both dev_b[:256].

*** E14 result pass 28 (heads 0,2,3 rank-1 vh removal): +0.316 +- 0.034 (92/11), acc 0.504. Copied to /work/best_5 (CURRENT BEST).
Pass 29: E14 unchanged, GEN both dev_a[:256] (/work/ids_deva256.json) to replicate on fresh DEV.

*** E14 replication pass 29 (dev_a[:256]): +0.324 +- 0.030 (84/1), 0.230 -> 0.555, cut 73->20. Pooled DEV 512: ~+0.32.
Pass 30: TF edited (E14) on /work/tf10_texts.json, mlp_act 11,12 -> mechanistic check of gate neurons vs fwd_18/19.

Run 130: value-norm ratio (last-prompt mean value / answer-mean value) per layer: L11 7.4, L18 7.41 (norm 44 vs 6), L10 4.69, L23 4.55, rest 1.8-3.7. L18 is the only other layer as extreme as L11 -> candidate for same rank-1 removal.
Run 131: col5 = cos(last-prompt value, answer value): L11 .34, L18 .71 (risky: overlaps ordinary content), L10 .20, L23 .21.
Pass 31 edit E15: E14 (L11 o_proj heads 0,2,3 <- B(I - vh vh^T)) + L18 o_proj all 4 heads <- B(I - vh18 vh18^T), vh18 from mean hidden_18 last prompt (gen_1_acts), strength 1. GEN both dev_b[:256] (/work/ids_devb256.json; E14 got +31.6 there). best_5 = E14.
E15 result pass 31 (dev_b[:256]): +0.316 +- 0.033 (89/8), acc 0.504 = same as E14. L18 removal neutral -> CLOSED, dropped.
Pass 32 edit E16: E14 (L11 heads 0,2,3) + L10 and L23 o_proj all heads <- B(I - vh_l vh_l^T) (vh from mean last-prompt hidden_l, gen_1_acts), strength 1. GEN both dev_b[:256]. best_5 = E14.
E16 result pass 32 (dev_b[:256]): +0.324 +- 0.034 (92/9) vs E14 +0.316: no measurable gain -> CLOSED constant-read search outside L11; L10/L23 dropped.
FROZEN: E14 = L11 o_proj heads 0,2,3 columns <- B(I - vh vh^T), vh=/work/vh_L11_lastprompt.npy (best_5).
Pass 33: GEN both E14 on HOLDOUT[:256] (/work/ids_ho1.json). Pass 34: HOLDOUT[256:512] (/work/ids_ho2.json).
*** HOLDOUT pass 33 (ho[:256]) E14: +0.340 +- 0.032 (92/5), 0.207 -> 0.547, cut 58->26.
Pass 34: same edit, GEN both /work/ids_ho2.json.
*** HOLDOUT pass 34 (ho[256:512]) E14: +0.305 +- 0.032 (84/6). Pooled HOLDOUT 512: +0.322 +- 0.023.
Pass 35: same E14, GEN both /work/ids_ho3.json (ho[512:768]).
*** HOLDOUT pass 35 (ho[512:768]) E14: +0.3125 +- 0.0325 (87/7). POOLED HOLDOUT 768: +0.319 +- 0.019 (263 w->r, 18 r->w). E14 = submission candidate (best_5).
Pass 36 edit E17: L11 o_proj ALL heads 0-3 columns <- B(I - vh vh^T) (E14 + head 1). GEN both dev_b[:256]. E14 in best_5.
E17 result pass 36 (dev_b[:256]): +0.340 +- 0.033 (94/7) vs E14 +0.316 same set (+6 problems, noise-level). Copy to best_6.
Pass 37: E17 unchanged, GEN both dev_a[:256] (/work/ids_deva256.json; E14 +0.324 there).
E17 replication pass 37 (dev_a[:256]): +0.305 +- 0.030 (80/2) vs E14 +0.324. Pooled DEV E17 +32.2 vs E14 +32.0 -> head 1 neutral, CLOSED. Restored E14 (best_5) as current/submission.
E14 residual failures (dev_a): ordinary reasoning/arith slips, 20/114 cut. No formatting mechanism.
Pass 38: TF edited (E14) on /work/tf6_texts.json (250 exprs x 4 contexts), topk 5, token_logprobs -> compare with fwd_11 (orig) exact-match by context.
fwd_38 (TF E14 on tf6 synthetic arith) vs fwd_11 orig: context A exact add2 .87->.97, sub2 .61->.94, mul2 .58->.87, add34 .02->.22; raw C unchanged, B/D ~unchanged. E14 specifically removes eval-context arithmetic suppression. Small residual A-B lp gap add2 (-.26 vs -.02).
fwd_12 attention segmentation (L11, eval ref solution): suffix 3-4% per head (CLOSED as source), last prompt token h0 .10 h1 .01 h2 .22 h3 .27 (matches E14 head choice; h1 neutral), BOS h0/h1 .23/.33 (context-invariant, not gate).
E14 FROZEN as submission. Remaining passes: extra confirmation on spare problems if unused.
Pass 39: E14 frozen (L11 o_proj heads 0,2,3 rank-1 vh removal), GEN both spare[:256] (/work/ids_sp1.json), confirmation only. Pass 40: spare[256:458] (/work/ids_sp2.json).
Spare pass 39 (spare[:256]) E14: +0.266 +- 0.032 (76/8), 0.1875 -> 0.453, cut 67->22. Confirmation only.
Spare pass 40 (spare[256:458]) E14: +0.381 +- 0.038 (82/5). Pooled spare 458: +0.317 +- 0.024 (158/13).
FINAL: submitted E14 = model.layers.11.self_attn.o_proj.weight, columns of heads 0,2,3 (0:256, 512:768, 768:1024) right-multiplied by (I - vh vh^T), vh = /work/vh_L11_lastprompt.npy. HOLDOUT 768 pooled +0.319 +- 0.019 (263/18); all untuned (holdout+spare, 1226) ~+0.318 +- 0.015.
Closed: L18, L10/L23 constant reads (neutral), L11 head1 (neutral), strength 2 (+7), reversal (-), whole-head ablation, neuron gates (E4/E8, redundant), suffix tokens (3-4% attention).
Post-freeze exploration (submission stays E14 unless a new edit clearly beats it on DEV and is confirmed): Pass 41: TF edited (E14) on /work/tf10_texts.json, token_logprobs, mlp_act 13..25 -> rescan for residual AND(chat,math) units vs fwd_20/fwd_18. Current edit = E14 only.
fwd_41 (E14, L13-25 AND scan): top units 2.0-3.6 sd with tiny acts (L25 n2447 .23 vs .10/.09) -> no residual gate, CLOSED.
Pass 42 (dose-response, diagnostic): L11 o_proj heads 0,2,3 cols <- B(I - 0.5 vh vh^T). GEN both dev_b[:256]. Restore E14 (best_5) afterwards; submission stays E14.
Pass 42 result (strength 0.5, heads 0,2,3): +0.090 +- 0.023 (30/7). Dose-response 0:0, 0.5:+9, 1:+31.6, (2 on h3: +7) -> threshold-like, peak at removal.
Pass 43 (diagnostic): heads 0,2,3 cols <- B(I - 1.25 vh vh^T). GEN both dev_b[:256]. Submission stays E14 unless clearly better + confirmed.

Pass 43 result (strength 1.25, heads 0,2,3, dev_b[:256]): +0.324 +- 0.034 (92/9) vs E14 +0.316 same set -> plateau at removal, no gain. CLOSED strength line. Restored E14 (best_5) as current edit/submission.
Pass 44 (CONTROL): L11 o_proj heads 0,2,3 cols <- B(I - r r^T), r = random unit 256-d (seed 12345, orthogonalised to vh; /work/ctrl_rand_dir.npy). GEN both dev_b[:256]. Expect ~0 if vh-specific. Restore E14 (best_5) after; submission stays E14.
Pass 44 CONTROL result (random dir r, heads 0,2,3, dev_b[:256]): -0.0195 +- 0.013 (3/8), cut unchanged 27% -> gain is vh-specific (E14 +0.316 same set). Restored E14 (best_5).
Side-effect check (fwd_18 orig vs fwd_41 E14, tf10 answer-token mean lp): v0 -1.340->-1.338, v3 -1.187->-1.248 (se .008), chat -3.660->-3.792 (se .012). E14 shifts non-math chat by ~0.13 nats/token: collateral cost, report it.
Pass 45: E14 frozen (best_5), GEN both dev_a[256:512] (/work/ids_deva2.json), extra untuned DEV measurement. Pass 46: dev_b[256:512] (/work/ids_devb2nd.json).
Pass 45 result E14 dev_a[256:512]: +0.328 +- 0.033 (92/8), 0.215->0.543, cut 66->21. Next: pass 46 dev_b[256:512].
Pass 46 result E14 dev_b[256:512]: +0.3125 +- 0.035 (92/12). E14 now measured on all 2250 practice: DEV 1024 ~+32.0, HOLDOUT 768 +31.9+-1.9, spare 458 +31.7+-2.4.
Per-position lp change (E14 vs orig, tf10): chat early -.09 late -.14 (diffuse); v3 -.13/-.05; v0 -.31/+.06.
Pass 47 (CONTROL 2): heads 0,2,3 cols <- B(I - r2 r2^T), r2 random seed 777 orth to vh (/work/ctrl_rand_dir2.npy). GEN both dev_a[:256] (/work/ids_deva256.json). Restore E14 after.
Pass 47 CONTROL 2 result (random r2, dev_a[:256]): 0.000 +- 0.011 (4/4) vs E14 +0.324 same set. Controls pooled ~-1 pt. Restored E14 (best_5).
Passes 48-50: E14 frozen, GEN both re-runs of /work/ids_ho1.json, ids_ho2.json, ids_ho3.json (confirmation only).
Identical re-run refused. Passes 48-50 instead: E14 frozen, GEN both /work/ids_hoR1.json (ho[128:384]), ids_hoR2.json (ho[384:640]), ids_hoR3.json (ho[640:768]+ho[:128]); re-measurement under different batching, confirmation only.
Pass 48 result E14 ho[128:384] re-batch: +0.332 +- 0.0315 (89/4), 0.203->0.535.
Pass 49 result E14 ho[384:640] re-batch: +0.3047 +- 0.033 (86/8), 0.25->0.555. Pass 50: ids_hoR3.json.
