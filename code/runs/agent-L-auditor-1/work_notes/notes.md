# NOTES
## Sets
DEV=/work/dev_ids.json (1350), HOLDOUT=/work/holdout_ids.json (900), seed 20261104 shuffle. HOLDOUT only for confirming a frozen candidate.
dev_a = first 512 of DEV.
## Current edit
none (original)
## Log
pass 1 (planned): generate original on dev_a, no edits.
## Plan
baseline, failure types, read failures.

pass 2 (planned): generate original on dev_a WITH activations (all layers 0..26, mlp true). no edits.
Findings so far: baseline dev_a acc 24.2%, cut 26.6%, failures are reasoning+loops not format. q_F*k_F const => norm gauge; scale of q,k,v/o,down are no-ops at run time. o_proj top dirs not shared across layers.

Findings: coord 1038 = 94-99% of residual energy, masked at pre_ff (w=-1), acts as RMS denominator/volume knob. near-constant neurons look natural. No dead neurons, no head imbalance.
Failure analysis: gold seen in only 26% of cut answers, 16.5% of finished-wrong: deficit is REASONING, loop fix max ~5.7 pts.
pass 3 (planned): teacher_forced ORIGINAL, no edit, 200 DEV problems (DEV[512:712]) x 5 contexts (eval,nosuf,qa,expert,raw), token_logprobs: test context gating of capability.

KEY LEAD: chat vs plain context, same ref tokens: result digits after '=' p=.833 eval vs .923 raw (+.138+-.013 logp); 67 tokens raw>.8&eval<.5 vs 0 reverse; dose: eval .32 expert .45 nosuf .59 raw .92 on those tokens. Own-answer arithmetic error 10.1% (1.7% in correct, 18% in cut). Likely context-gated arithmetic suppression.
pass 4 (planned): teacher_forced ORIGINAL no edit, 120 DEV[512:712] problems x (eval, raw) /work/pair_texts.json, logprobs+topk5+hidden all layers.

Chat wrong digits: on deficit tokens mass moves to OTHER DIGITS (.47 vs .06), computation disturbed, not format.
Depth profile: eval-vs-raw paired shift jumps at hidden_12 (= output of model.layers.11, full attn) 0.124->0.365. Candidate mode writer. Layer 11 k_norm mean 1.2, o_proj top dir head3 78%.
pass 5 (planned): teacher_forced ORIGINAL, 2 texts (pair p=12) attn_layers 5,11,17,23: see what heads read.

Attention: layer11 heads 2,3 (and 0) read token 79 '\n' after 'model' (last prompt token) throughout answer (.23,.31,.11); layer17 heads 1,2,3 read '\n' after 'user'. Raw twin: no anchor, BOS instead.
EDIT E1 (pass 6): layers.11 o_proj head 2,3 columns zero (strength 1). Screen: teacher_forced EDITED /work/pair_texts.json logprobs. Compare via score_pair.

pass 6 result E1: eval res p .820->.891, raw .910->.897 (deficit closes); res dlogp eval +.059(.026) raw -.060; nondigit dlogp eval -.187 raw -.237 (general cost).
pass 7 (planned): generate EDITED (E1) on dev_a 512, compare to pass 1 baseline 24.2%.

pass 7 result E1 (zero L11 o_proj heads 2,3) dev_a: acc .2168 vs baseline .242 (unpaired), cut .0098 vs .266, mean tokens 160.6. NO accuracy gain; cut-offs gone. Analysing paired with gen_1.

E1 analysis: paired vs gen_1 -2.5+-2.2; marker .26->.059, 'Let $X$' style gone, cut .27->.01, arith err (crude) finished .196->.126. Extraction not the issue. E1 saved in /work/e1 (NOT a good candidate).
VOLUME LEAD: coord 1038 grows 0->15000, masked at pre_ff (1+w=0 from L3) and model.norm (1+w=-.016) so acts via RMS denominators. At res tokens L11 writes +1461 eval vs +892 raw; signal/volume ratio eval/raw ~.85 at L24-26 (res), .92-.97 (oth). L11 o_proj row1038 3.12x median (h3 2.44), row ~1% of output energy so zeroing does not disturb RMS. Across problems volume barely predicts correct (|r|<=.2 finished).
Current edit V1: model.layers.11.self_attn.o_proj.weight row 1038, columns of heads 2,3 (512..1023) = 0, nothing else. E1 dropped.
pass 8 (planned): teacher_forced EDITED V1, /work/pair_texts.json, logprobs + hidden all 0..26. Predict: if volume is E1's active part, eval res p ->~.89 with less non-digit cost; else refuted.

Results since pass 8: V1 (L11 row1038 h2,h3=0) screen: volume -18% at L12 (gain +26%, decays to +4% at output), L12 eval/raw ratio .94->1.015 but late ratio .865->.874 unchanged; res dlogp eval +.025(.006), raw -.006, nondigit -.018. p>.5 res eval .863 -> .871 (E1 .905, raw .919).
Careful arithmetic: E1 halves expr errors (10.1->5.2%; both-finished 6.8->3.6) but error-free finished answers only 31% correct: reasoning/setup is binding, arithmetic is not. Loops begin ~15% in, before any final answer.
E1 split: rescues orig-cut (~+3.9 pts), damages orig-finished (~-6.4 pts).
NOISE FLOOR: gen_1 and gen_2 identical on all 512 => edited-only dev_a pass (same order) is exactly paired with stored baseline.
CLOSED: isolated rank-1 outlier in o_proj/down_proj (spectra smooth across layers); single extreme neuron (62 |mean|>.8 units, shared cut/length mode r~+-.75, partial r with correct falls to ~.1-.2, top .33 vs noise .25); volume as main cause (only ~1/4 of E1 digit gain); easy-problem anomaly (step count unreliable, failures look ordinary); extraction/format.
L12 eval-raw mean shift: 70% of energy constant, spread (85 coords for 50%), 60% in head3 writable subspace vs .22 random.
Current edit E1s50: model.layers.11.self_attn.o_proj.weight columns of heads 2,3 (512..1023) x 0.5, nothing else.
pass 9 (planned): generate EDITED E1s50 on dev_a (512, /work/dev_a.json), compare with gen_1 exact paired. Question: do cut-rescue and reasoning-damage separate?
Plan: strengths then head3-only/head2-only; direction removal of L12 shift; other anchor heads (L17 h1-3, L23 h3).

pass 9 result E1s50 (L11 h2,h3 x .5) dev_a: acc .2988 vs .2422, cut .133. See cmp_gen. Selected on dev_a, needs independent confirmation.

pass 9 detail E1s50: +5.66 (se 1.60) paired vs gen_1, w->r 49 r->w 20, cut .266->.133, marker .26->.371, orig-cut +2.54 pts, orig-fin +3.12 pts, fifths all positive (2.9..7.8). saved /work/best_2. Dose non-monotonic: keep 100% 24.2, 50% 29.9, 0% 21.7.
Current edit E1s75: model.layers.11.self_attn.o_proj.weight columns of heads 2,3 x 0.75, nothing else.
pass 10 (planned): generate EDITED keep75 on dev_a; then keep25; then head3-only, head2-only.

pass 10 result keep75 (L11 h2,h3 x .75) dev_a acc .2617 cut .201.
Current edit keep25: model.layers.11.self_attn.o_proj.weight columns of heads 2,3 x 0.25, nothing else.
pass 11 (planned): generate EDITED keep25 on dev_a.

pass 11 result keep25 (L11 h2,h3 x .25) dev_a acc .2871 cut .057. Dose: keep100 24.2, 75 26.2, 50 29.9, 25 28.7, 0 21.7; cut 26.6,20.1,13.3,5.7,1.0.
Current edit H3k40: model.layers.11.self_attn.o_proj.weight columns of head 3 only (768..1023) x 0.4, heads 0,1,2 untouched.
pass 12 (planned): generate EDITED H3k40 on dev_a. Then head2-only.

pass 12 result H3k40 (L11 head 3 only x .4) dev_a acc .3379 cut .066. saved /work/best_3.
Current edit H3k0: model.layers.11.self_attn.o_proj.weight columns of head 3 (768..1023) = 0, heads 0,1,2 untouched.
pass 13 (planned): generate EDITED H3k0 on dev_a. Then head-3 strengths .2/.6, head2-only control.

pass 13 result H3k0 (L11 head 3 zero) dev_a acc .3242 cut .018. head 2 removal is what damaged E1.
Current edit H3k20: model.layers.11.self_attn.o_proj.weight columns of head 3 x 0.2, heads 0,1,2 untouched.
pass 14 (planned): generate EDITED H3k20 on dev_a. Then k60, head2-only control, then other heads.

pass 14 result H3k20 (L11 head 3 x .2) dev_a acc .3828 cut .066; saved /work/best_4. Head-3 dose: k100 24.2, k40 33.8, k20 38.3, k0 32.4. Peak likely inflated by selection.
Current edit H3k20: model.layers.11.self_attn.o_proj.weight columns of head 3 x 0.2, heads 0,1,2 untouched.
dev_b = next 256 of DEV after dev_a, /work/dev_b.json (tuning, not holdout).
pass 15 (planned): generate BOTH H3k20 on dev_b (256 per model).

pass 15 result BOTH H3k20 on dev_b: orig .1797 edited .3047 paired +12.5 se 2.93 (46 w->r,14 r->w). dev_a: +14.06 se 2.21. head 2 is needed, head 3 harmful, k20 > k40 (+4.49 se 1.75), k20 > k0 (+5.86 se 1.92). Best so far H3k20 in /work/best_4.
Next: mechanism of head 3 (top direction / anchor), other harmful heads, HOLDOUT only after freeze.

Head-3 structure: anchor value 99.7% constant; head 3 write of anchor mean is 88.5% on coord 1038 (gain adjusted) vs h0 79%, h1 31%, h2 16%. Head 3 top singular dir (S 6.88, 18.7% energy) NOT aligned with anchor value (cos .025); anchor goes through dir 2 (cos .33, 47% of write).
Current edit V3: model.layers.11.self_attn.o_proj.weight row 1038, head 3 columns (768..1023)=0 only. Nothing else.
pass 16 (planned): generate EDITED V3 on dev_a. Tests volume write alone vs whole-head k20 (+14.06).

pass 16 result V3 (row1038, head 3 zero) dev_a acc .2227 cut .232: volume write is NOT the harm. Line closed.
Current edit T1: model.layers.11.self_attn.o_proj.weight, head 3 columns: top right singular direction of those columns removed (O3 := O3(I - v1 v1^T)), a=1. Heads 0-2 untouched.
pass 17 (planned): generate EDITED T1 on dev_a.

pass 17 result T1 (head 3 top singular dir removed) dev_a acc .2266 cut .320: top dir not harmful, removing raises loops. Note: post-attn RMS means only relative head weight matters.
Anchor value 99.7% constant was partly artefact of common coord-1038 component; use anchor-specific diff du = mean(v(anchor) - v(mean answer)), L11 v_proj on stored hidden.
Current edit T2: model.layers.11.self_attn.o_proj.weight head 3 columns: O3 := O3(I - du du^T), a=1. Heads 0-2 untouched. du saved /work/du_anchor_diff.pt
pass 18 (planned): generate EDITED T2 on dev_a. Decisive test of anchor-broadcast story.

pass 18 result T2 (head 3 du direction removed, a=1) dev_a acc .4629 cut .092; saved /work/best_5. HUGE vs H3k20 .3828 and baseline .2422.
Current edit T2 unchanged. pass 19 (planned): generate BOTH T2 on dev_b (fresh for this edit, 256 per model).

pass 18 detail T2 dev_a paired vs gen_1: +22.07 se 2.30, w->r 138 r->w 25, cut .266->.092, marker .26->.129, toks 344->262; orig-cut +9.18 pts, orig-fin +12.89 pts; fifths 14.7..27.5; T2 vs H3k20 +8.01 se 2.12. Endings boxed 303 marker 153: grader falls back to last number, fine.
Current edit T2: model.layers.11.self_attn.o_proj.weight head 3 columns (768..1023): O3 := O3(I - du du^T), a=1, du=/work/du_anchor_diff.pt. Heads 0-2 untouched.
pass 19 (planned): generate BOTH T2 on /work/dev_b.json (256 per model).

pass 19 result BOTH T2 dev_b: orig .1797 edited .4531 paired +27.34 se 3.25 (79 w->r, 9 r->w), cut .258->.133. Confirms T2 far above H3k20 (+12.5) on data unused for T2. Best so far T2 in /work/best_5.
Next: map anchor reads at L5/L17/L23 and L11 other heads; then strengths; reserve 4 passes for HOLDOUT.

Anchor map (fwd_5 attn x fwd_4 values, mean template values): BOS is default sink everywhere both formats. Chat-only reads are template newlines: L11 h3/h2/h0 nl_m (.307/.234/.112), L17 h1/h2/h3 nl_u (.358/.148/.139), L23 h3/h0/h1 nl_e (.106/.03/.03). Values: L11 sot1 |v| 136 (L14 105, L18 75, L18 nl_e 79) but sot1 mass ~.003. Written size |O_h m v| (includes row 1038): L11 h3 nl_m 6.77 >> L11 h3 sot1 1.93, L23 h3 nl_e 1.85, L11 h2 nl_m 1.54, L11 h0 nl_m 1.06, L17 h1 nl_u .82. L18 is windowed: anchor value 44 vs 6, |O du|/frob ~.25 vs .07 typical, attention not observable.
dev_c = DEV minus dev_a, dev_b (582 problems), unused so far.
Current edit T2 (unchanged): model.layers.11.self_attn.o_proj.weight head 3 columns O3 := O3(I - du du^T), a=1.
pass 20 (planned): teacher_forced EDITED T2, /work/pair_texts.json, logprobs, topk 5, hidden 0..26. Compare eval-raw depth profile with fwd_4: does the L12 jump close, does a later layer add a new one?
Candidates after that: T2 + du removal on L11 h0,h2; L18 anchor removal; L23 nl_e removal; L17 nl_u removal; strength of T2.

pass 20 result T2 teacher_forced pairs: res p eval .820->.909 (raw .910->.922), nondigit eval .628->.653 (raw .647->.652), dlogp res +.122(.017). L12 shift .365->.204, no new jump later; increment at L12 .331->.188. Removed vector cos -.80 with orig shift; remaining shift proj on removed .10 (a~1 is nearly exhausted). Remaining increment aligns with du write via h2 (+.54) and h0 (+.32), h1 +.15.
Current edit T3: model.layers.11.self_attn.o_proj.weight, heads 0,2,3 columns: O_h := O_h(I - du du^T), a=1, same du. Head 1 untouched.
pass 21 (planned): generate EDITED T3 on dev_a (512), exact pair with gen_18 (T2) and gen_1 (orig).

pass 21 result T3 (L11 heads 0,2,3 du removed) dev_a acc .5391 cut .084; saved /work/best_6. T2 .4629, orig .2422.

pass 21 detail T3 vs orig +29.69 se 2.27 (166/14); vs T2 +7.62 se 2.05 (76/37); fifths vs T2 11.7,-1.0,17.6,4.9,4.9. orig-cut .022->.426, orig-fin .322->.580. Styles Let 380 acc .582, Heres 132 acc .417. Crude arith err no longer separates correct (.584 vs .577). Steps 1-2 .693, 3 .544, 4 .451, 5+ .322. Failures look like setup errors.
Current edit T3 (unchanged): model.layers.11.self_attn.o_proj.weight heads 0,2,3 O_h(I-du du^T), a=1.
pass 22 (planned): teacher_forced EDITED T3, /work/pair_texts.json, logprobs, topk 5, hidden 0..26. Depth profile, remaining L12 increment alignment.

pass 22 result T3 teacher_forced pairs: L12 mean shift res .365->.120 (below L11 .124), output shift .191->.106; res p eval .918 raw .923, digit_other eval .854 raw .828, nondigit eval .660 raw .653. Chat deficit closed. Remaining L12 increment aligns with nothing (|cos du writes|<.06), remaining proj on removed .036: L11 exhausted, strength>1 not indicated.
Fingerprint scan (cos of layer increment with gain write of template value through a head): L10 h1 nl_m +.53 (inc .149, largest), h2 +.22, h0 -.36; L4 h2 +.67 h3 +.41; L7 h2 +.55; L8 h1 +.38; L13 h1 +.23; L15 h1 +.34; L21 h1 nl_e +.69; L22 h0 nl_e +.33; L23 h3 nl_e +.68; L25 h2 +.38 h3 +.34. Alignment alone is not harm (coord 1038, T1 precedents).
Current edit T4: T3 (L11 o_proj heads 0,2,3 du removed, a=1) PLUS model.layers.10.self_attn.o_proj.weight head 1 columns (256..511) O(I - du10 du10^T), du10 = /work/du10.pt from generation activations.
pass 23 (planned): generate EDITED T4 on dev_a, exact pair with gen_21 (T3).

pass 23 result T4 (T3 + L10 h1 du10) dev_a acc .5859 cut .119, saved /work/best_7. T3 .5391 cut .084.

pass 23 detail T4 vs orig +34.38 se 2.41 (194/18); vs T3 +4.69 se 1.92 (61/37), fifths -1.9,12.6,-1.0,5.9,7.8. Cuts 43->61: 40 new (47.5% were right in T3), 22 gone (77% now right). 65% of cuts are literal repeats, 44% contain gold. Style Let .63 Heres .47.
Looping is now a second failure: head-3 whole scaling removed cuts (k0 1.8%, k20 6.6%), so plan test T4 + head 3 scale later.
Current edit T5: T4 (L11 h0,h2,h3 du; L10 h1 du10) PLUS L4 o_proj h2,h3 du4; L7 h2 du7; L8 h1 du8, a=1. DU in /work/DU_nlm.pt (recipe of block 60 per layer). build() in kernel rebuilds from originals.
pass 24 (planned): generate EDITED T5 on dev_a, pair with gen_23 (T4).

pass 24 result T5 (T4 + L4 h2,h3, L7 h2, L8 h1 nl_m du) dev_a acc .5645 cut .109: no gain over T4 .5859. Early group CLOSED, reverted. High cos on small increments is weak evidence.
Current edit T6: T4 (L11 h0,h2,h3 du; L10 h1 du10) PLUS nl_e direction (mean v(nl_e)-mean v(ans), 120 texts, /work/DU_nle.pt) removed from L21 h1, L23 h3, L25 h2,h3, a=1.
pass 25 (planned): generate EDITED T6 on dev_a, pair with gen_23 (T4). Then mid group L13 h1, L15 h1; L10 h2; head 3 scale for loops.

pass 25 result T6 (T4 + nl_e removal L21 h1, L23 h3, L25 h2,h3) dev_a acc .5918 cut .107: tie with T4 .5859. Late group reverted unless paired shows real effect.
Only L10 and L11 cuts have improved accuracy so far.
Current edit T7: T4 (L11 h0,h2,h3 du removed; L10 h1 du10 removed) PLUS L11 head 3 columns x 0.5.
pass 26 (planned): generate EDITED T7 on dev_a, pair with gen_23 (T4). Question: is a loop driver left in head 3 after du removal?

pass 26 result T7 (T4 + L11 head 3 x .5) dev_a acc .5586 cut .070, wrong up 155->190: rest of head 3 is useful, line CLOSED. T6 vs T4 +.59 se 1.28: late group CLOSED.
Current edit T4 restored (L11 h0,h2,h3 du; L10 h1 du10).

Natural experiment on 6 dev_a variants: mixed-style problems 56; within-problem Let-Heres acc +.074 se .048, cut +.090 se .034: style not a target. Union acc .754, never solved .246, always solved .322; 41% of T4 failures solved by some variant (near misses, flips).
Current edit T4 (unchanged): L11 o_proj h0,h2,h3 du, L10 o_proj h1 du10, a=1.
pass 27 (planned): teacher_forced EDITED T4, /work/pair_texts.json, logprobs, topk 5, hidden 0..26. Depth profile after L10 cut, remaining fingerprints.

pass 27 result T4 teacher_forced pairs: L10 inc .149->.119, no strong alignment left at L10/L11; res eval .920, digit_other .855, nondigit .658. Teacher-forced screen saturated (T3->T4 barely moves, generation +4.7): rank only by generation now.
Queue: T8 mid group; sot1 direction L11; L18 template direction; L11 h2,h3 gain up; L10 h2,h3.
Current edit T8: T4 (L11 h0,h2,h3 du; L10 h1 du10) PLUS L13 h1 DU[13], L15 h1 DU[15], a=1.
pass 28 (planned): generate EDITED T8 on dev_a, pair with gen_23 (T4).

pass 28 result T8 (T4 + L13 h1, L15 h1 du) dev_a acc .5840 cut .121: tie, mid group CLOSED.
L18 structure: template dirs nearly collinear (cos .83-.93 vs .2-.3 elsewhere), heads pass it near top singular value.
Current edit T10: T4 PLUS model.layers.18.self_attn.o_proj.weight all four heads O(I - d18 d18^T), a=1, d18=/work/d18.pt (top singular vector of template value diffs).
pass 29 (planned): generate EDITED T10 on dev_a, pair with gen_23 (T4).

pass 29 result T10 (T4 + L18 all heads d18 removed) dev_a acc .5645 cut .125: below T4 .5859, reverted.
Current edit T4 restored (L11 h0,h2,h3 du; L10 h1 du10).

pass 29 detail T10 vs T4 -2.15 se 1.37 (19/30), fifths 4.9,-2.0,-1.0,-5.9,-6.8: L18 CLOSED.
READ-SIDE LEAD (no edit before touched q/k). L11 key stats on fwd_27 eval texts: anchor key diff vs mean answer key consistent (cos .93, min .87), cos .74 with BOS key diff. Margins (logit, mean q): anchor h0 4.88 h1 6.16 h2 5.54 h3 4.39; BOS 6.00 9.30 3.48 3.58. Anchor-specific part (orthogonal to BOS diff) h0 1.37 h1 .71 h2 3.51 h3 2.29. up, ur=qw*up normalized in /work/q11_dirs.pt.
Current edit T11: T4 (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1) PLUS model.layers.11.self_attn.q_proj.weight rows of h0,h2,h3: Q_h := (I - ur ur^T) Q_h, a=1. Predicted h2 anchor margin 5.54->2.04, h3 4.39->2.10, BOS unchanged. Head 1 untouched.
pass 30 (planned): generate EDITED T11 on /work/dev_a.json, pair with gen_23 (T4). If tie, one attn pass to check mass moved before closing.
Queue after: L17 h1,h2,h3 nl_u (never tested); L11 sot1; L10 h2,h3; L11 gain h2,h3; confirm T4 on dev_c. Reserve 4 passes for HOLDOUT.

pass 30 result T11 (T4 + L11 q_proj h0,h2,h3 up removed) dev_a acc .5586 cut .1055, wrong 175 vs T4 155.
Current edit T11 unchanged. pass 31 (planned): teacher_forced EDITED T11, /work/attn_texts.json, token_logprobs, attn_layers [11]: did anchor mass move, and to where? Compare with fwd_5.

pass 30 detail T11 vs T4 -2.73 se 1.74 (33/47), both finished -4.23 se 1.78; cuts gone 32 (13 right), new 25.
pass 31 result attention L11 orig->T11, eval text: anchor h0 .112->.036 h2 .234->.011 h3 .307->.046; mass moved to prompt+prev+BOS, profile now like raw; raw control unchanged. Valid test, hypothesis REFUTED: harm was the anchor WRITE, the read is harmless once muted. Read-side CLOSED. L11 gain-up CLOSED (T11 already gave h2,h3 about 25% more content mass, no gain).
BIAS WARNING: T4+X variants on dev_a average .574 vs T4 .586; T4 won selection on dev_a so comparisons there lean against X. Screen on fresh data from now on.
dev_c512 = first 512 of dev_c, /work/dev_c512.json, unused by any edit.
Current edit T4 restored (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1), q_proj dropped, identical to best_7.
pass 32 (planned): generate EDITED T4 on /work/dev_c512.json. pass 33 (planned): generate EDITED T3 on same, exact pair: unbiased L10 check.
Queue: L17 h1,h2,h3 nl_u (never tested); L11 sot1 direction (write size h3 1.93 > h2 nl_m 1.54); L10 h2; loop analysis of T4 cuts. Reserve 4 passes for HOLDOUT (both, split 256).

pass 32 result T4 on dev_c512 acc .5312 cut .1367 wrong 172 (fresh data, gen_32).
Current edit T3: model.layers.11.self_attn.o_proj.weight heads 0,2,3 O_h(I-du du^T), a=1 only (L10 edit dropped, same as best_6). T4 stays in /work/best_7.
pass 33 (planned): generate EDITED T3 on /work/dev_c512.json, exact pair with gen_32.

pass 33 result T3 on dev_c512 acc .5020 cut .0977 wrong 206. T4 vs T3 fresh +2.93 se 1.78 (49/34), fifths 2.0,-2.0,2.9,2.9,8.7; both finished +2.85 se 1.84. L10 cut CONFIRMED positive though smaller than dev_a +4.69. New cuts 81 over both sets, 29 right in T3: at most ~2.8 pts.
T4 cuts analysed (131 over dev_a+dev_c): 6 correct, no final marker before loop, loop start median 20% of text, 41% strictly periodic, recheck-dominated about 1/3. Rambling, no single behaviour. Not a site.
Write-vector scan: L11 heads and L10 h1 writes mutually near orthogonal (|cos|<=.21), combined L11 vs L10h1 .17: NO shared target direction. Early-layer capacity is smooth with depth, MLP down_proj no isolated writer (max neuron cos ~.34, smooth). CLOSED.
Current edit T12: T4 (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1) PLUS model.layers.17.self_attn.o_proj.weight h1,h2,h3 O(I-d17 d17^T), a=1, d17 = dir_for(17,'nl_u').
pass 34 (planned): generate EDITED T12 on /work/dev_c512.json, pair with gen_32 (T4).

pass 34 result T12 (T4 + L17 h1,h2,h3 nl_u) dev_c512 acc .5430 cut .127: vs T4 +1.17 se 1.27 (24/18), tie; both finished +.48 se 1.17. L17 not adopted (8.6% tensor change, cuts part of steady answer write).

sot1 at L11: ds |cos| .72 with h3 top singular vector (T1 direction), .50 h0; attention flat (h3 mean .0028 max .0088, no digit preference), 23% tensor change. CLOSED without a pass, T1 precedent argues against.
L10 write structure: in du10 direction h0 opposes h1,h2,h3 (cos -.53,-.49,-.36).
Current edit T13: T4 PLUS model.layers.10.self_attn.o_proj.weight heads 0,2,3 O(I-du10 du10^T), a=1, i.e. du10 removed from all four L10 heads; L11 o_proj h0,h2,h3 du unchanged.
pass 35 (planned): generate EDITED T13 on /work/dev_c512.json, pair with gen_32 (T4).
Best remains T4 in /work/best_7. Reserve 4 passes for HOLDOUT.

pass 35 result T13 (T4 + L10 h0,h2,h3 du10, so all 4 L10 heads) dev_c512 acc .5234 cut .0703 wrong 210 tokens 237: vs T4 -.78 se 1.92 (46/50). Saved only in notes (build spec), NOT adopted.
T13 split: 131 problems switched Let->Heres, paired -13.74 se 4.57 (10/28); 381 same-style +3.67 se 1.99 (36/22) (post-treatment subset, hypothesis only). Style shares T4 Let 353/Heres 159, T13 222/290. Within style acc T4 Let .567 Heres .453, T13 .590 .472. Cut Let .193->.144.
Scope note: will NOT bias the opening directly (global output steering). Only ask whether style switch and reasoning gain live in different heads.
Current edit T14: T4 PLUS model.layers.10.self_attn.o_proj.weight heads 2,3 O(I-du10 du10^T), a=1 (L10 h1,h2,h3 cut, h0 untouched); L11 o_proj h0,h2,h3 du unchanged.
pass 36 (planned): generate EDITED T14 on /work/dev_c512.json, pair with gen_32 (T4), gen_35 (T13). Judge by total accuracy; then h0-only if needed.

pass 36 result T14 (T4 + L10 h2,h3 du10; h0 untouched) dev_c512 acc .5215 cut .1094 wrong 191.

pass 36 detail T14 vs T4 -.98 se 1.42 (24/29), only 5 style switches, same-style -.99, cut .137->.109. L10 h2,h3 CLOSED. Style switch of T13 is L10 h0 du10 alone (causal, single head direction). h0-only not tested: switch cost comes with it. Reversing it to favour Let NOT done: that is steering answer style, not repair.
Switched problems: Let 343 tokens acc .588, Heres 162 tokens acc .450; Heres failures are hasty setup errors, not arithmetic.
Current edit T4 restored (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1), identical to best_7.
pass 37 (planned): teacher_forced EDITED T4, /work/own_texts.json (meta /work/own_meta.json): T4 own finished answers, 4 groups Let/Heres x right/wrong, 85 each, contexts eval,nosuf,raw, token_logprobs, topk 5. Question: residual context gate specific to Heres mode? Compare across groups (generated under eval, so eval is favoured by construction).

pass 37 result own-answer screen (T4, 340 answers x eval,nosuf,raw): res digit logp eval~raw in all groups (-.028/-.031 Heres wrong); eval-only tokens per 100: Heres wrong 2.9 right 3.3, Let wrong .58 right .58. Wrong answers carry no extra context signal. Failures are confident setup errors shared by raw. Heres-mode gate CLOSED. Cut rescue also closed as target (rescued cuts right only 31% T13, 19% T14).
Next: check du drift under T4 at L11 (fwd_27), then dose on L11 and L10 h1 at dev_c512, then freeze and HOLDOUT.

du drift check under T4 (fwd_27): cos(dT,du) .9965, residual .7% energy; anchor write mass*|gO d| orig h0 10.5 h2 20.0 h3 89.6, T4 1.6, 3.2, 7.4. Direction dTh saved /work/du_T4.pt.
Current edit D1: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=.5 (dose test of main mechanism).
pass 38 (planned): generate EDITED D1 on /work/dev_c512.json, pair with gen_32 (T4). Predict accuracy clearly below T4.
Then dose: L11 a=1.25; L10 h1 a=.5, 1.5; T4prime. Then freeze, HOLDOUT (4 passes). 

pass 38 result D1 (L10 h1 a=1, L11 h0,h2,h3 du a=.5) dev_c512 acc .3750 cut .1562 wrong 242, vs T4 .5312. Strong dose response for main mechanism; accuracy steep near a=1, so residual write may matter.
Current edit D2: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1.2 (overshoot, write along du = -.2 of original).
pass 39 (planned): generate EDITED D2 on /work/dev_c512.json, pair with gen_32 (T4). Locates the null: fall = null near 1, rise = stopped short.

pass 39 result D2 (L11 h0,h2,h3 du a=1.2, L10 h1 a=1) dev_c512 acc .5527 cut .0957 wrong 181: vs T4 +2.15 se 1.82 (49/38), both finished 0.0 se 1.91, cuts gone 38 (16 right) new 17 (4 right in T4). Selected on dev_c512, needs independent check. Let .614 Heres .382; 23 Heres->Let. Dose so far a=.5 .375, 1 .531, 1.2 .553. D1 vs T4 -15.62 se 2.15 (27/107).
BOS-match prediction REFUTED: a* bos 1.04, no template token above; overshoot has no derived mechanism, strength is empirical. Heres share follows a: .45,.31,.27. Report style side effect; if gain at higher a is only style shift, count against.
Current edit D3: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1.5 (write along du = -.5 of original).
pass 40 (planned): generate EDITED D3 on /work/dev_c512.json, pair with gen_32 (T4), gen_39 (D2).
Then independent check of winner on dev_a (pair gen_23), then L10 h1 a>1, T4prime residual, then freeze and HOLDOUT 4 passes.

pass 40 result D3 (L11 h0,h2,h3 du a=1.5, L10 h1 a=1) dev_c512 acc .4922 cut .1035 wrong 209: vs T4 -3.91 se 1.91 (38/58), vs D2 -6.05 se 1.87 (31/62); both finished vs D2 -4.85 se 1.99. Within Let acc .536 (D2 .614), Heres .401. Dose a=.5 .375, 1 .531, 1.2 .553, 1.5 .492: optimum between 1 and 1.5, a=1.5 REJECTED.
CORRECTION: Heres share .45,.31,.27,.33 at a=.5,1,1.2,1.5: not monotone, earlier note "follows a" held on 3 points only.
Kernel note: block 120 overwrote helpers pair (now returns mean,se,up,down), rd, fin (now a function); sty unchanged. best_8 was empty after block 119, D2 had not been saved.

Current edit D2: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1.2, nothing else. Saved /work/best_8. T4 stays in best_7.
pass 41 (planned): generate EDITED D2 on /work/dev_a.json, exact pair with gen_23 (T4). dev_a leans against D2 (T4 won there), dev_c512 leans for it (D2 won there); read pooled.
Decision rule fixed before the result: D2 replaces T4 only if pooled paired gain over 1024 is positive by at least one SE. Else T4.
Open: was L11 h1 ever tested along du? Overshoot on h0,h2,h3 may be offsetting h1 residual write.

pass 41 result D2 dev_a acc .5801 cut .1055 wrong 164: vs T4 -.59 se 1.80 (41/44). POOLED dev_a+dev_c512 D2-T4 +.78 se 1.28 (90/82), below the one-SE rule: T4 STAYS. D2 reported as tie. Cuts fell on both sets (.119->.105, .137->.096), finished answers gave it back (-1.41 se 1.69, 0.0 se 1.91). L11 strength: plateau a=1 to 1.2, fall at 1.5. Consistent with pass 22 note "strength>1 not indicated".
Current edit L1: L10 o_proj h1 du10 a=1.5 (write along du10 = -.5 of original) PLUS L11 o_proj h0,h2,h3 du a=1 (L11 same as T4). T4 in best_7, D2 in best_8.
pass 42 (planned): generate EDITED L1 on /work/dev_c512.json, pair with gen_32 (T4, a=1), gen_33 (T3, a=0). Inside +-1.8 counts as tie, T4 stays.
Then: rank-2 residual on L11 (du + rh from /work/du_T4.pt, h0,h2,h3), then freeze, HOLDOUT 4 passes. Free passes after HOLDOUT reserve: 5 before pass 42.

pass 42 result L1 (L10 h1 du10 a=1.5, L11 h0,h2,h3 du a=1) dev_c512 acc .5469 cut .1445 wrong 160: vs T4 +1.56 se 1.46 (32/24), vs T3 +4.49 se 1.87 (58/35); both finished +1.20 se 1.38; 4 style switches, same-style +1.57; Let .602 Here .429. TIE by the +-1.8 band. L10 dose a=0 .502, 1 .531, 1.5 .547. L1 saved /work/best_9.
L11 anchor read orig eval text: mass [0.112, 0.022, 0.234, 0.307], |O_h du| [0.628, 0.303, 0.418, 1.383], mass*write [0.07, 0.007, 0.098, 0.424].
Current edit L2: L10 o_proj h1 du10 a=2 (write along du10 = -1 of original) PLUS L11 o_proj h0,h2,h3 du a=1 (L11 same as T4). T4 best_7, D2 best_8, L1 best_9.
pass 43 (planned): generate EDITED L2 on /work/dev_c512.json, pair with gen_32 (T4), gen_42 (L1). Fall: bracket, send a=1.5 to dev_a. Rise: send a=2 to dev_a. L10 strength changes only if pooled gain over T4 on 1024 is >= +1 SE.
Then rank-2 residual if a pass is free. Reserve 4 passes for HOLDOUT.

pass 43 result L2 (L10 h1 du10 a=2, L11 h0,h2,h3 du a=1) dev_c512 acc .5156 cut .1777 wrong 162. L10 dose a=0,1,1.5,2: acc .502,.531,.547,.516; cuts rise throughout, wrong answers 206,172,160,162. a=2 REJECTED, peak bracketed between 1 and 2.
L11 h1 CLOSED without a pass: anchor mass .022, mass*write .007 vs h3 .424.
Current edit L1: L10 o_proj h1 du10 a=1.5 PLUS L11 o_proj h0,h2,h3 du a=1, identical to best_9. T4 best_7, D2 best_8.
pass 44 (planned): generate EDITED L1 on /work/dev_a.json, exact pair with gen_23 (T4). dev_a leans against L1, dev_c512 leans for it; read pooled. L1 replaces T4 only if pooled gain over 1024 >= +1 SE.
Candidate idea after that: L1 + D2 strength (L11 a=1.2), since D2 lowers cuts and L1 lowers wrong answers; or rank-2 residual. Reserve 4 passes for HOLDOUT, 7 left before pass 44.

pass 44 result L1 dev_a acc .5703 cut .1387 wrong 156: vs T4 -1.56 se 1.61. POOLED dev_a+dev_c512 L1-T4 0.00 se 1.09, below one-SE rule: T4 STAYS. L1 tie. Wrong-answer drop did not replicate. L1+D2 combination CLOSED (its premise failed).
Current edit R2: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1 and rh a=1 (rank-2, rh = residual of anchor diff under T4, /work/du_T4.pt, orthogonalised to du). T4 best_7, D2 best_8, L1 best_9.
pass 45 (planned): generate EDITED R2 on /work/dev_c512.json, pair with gen_32 (T4). Above +1.8: pass 46 on dev_a. Otherwise freeze T4. Then HOLDOUT, 4 passes both, 256 per model, all 900.

R2 NOT RUN: rel change vs T4 was 6.2%, not ~.8%. rh is a high-gain direction (|O_h rh| ~1.1 on h3); removing it cuts a whole channel from all tokens, the T1/T10 failure pattern. My reason for rank-2 over T4prime was wrong: T4prime nulls dTh, the whole anchor diff under T4, nothing comes back for the anchor. L1 pooled exact: 0.00 se 1.09 (62/62).
Current edit T4prime: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 O(I-dTh dTh^T) a=1, dTh from /work/du_T4.pt (replaces du). T4 best_7, D2 best_8, L1 best_9.
pass 45 (planned): generate EDITED T4prime on /work/dev_c512.json, pair with gen_32 (T4). Above +1.8: pass 46 on dev_a. Otherwise freeze T4. Then HOLDOUT, 4 passes both, 256 per model, all 900.

pass 45 result T4prime dev_c512 acc .5352 cut .1504 wrong 164: vs T4 0.39 se 1.49 (30/28). Tie, leftover anchor write not limiting. Line CLOSED.
FROZEN CANDIDATE: T4 (/work/best_7): L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1. D2 and L1 ties, T4prime tie.
Current edit LO: L10 o_proj h1 du10 a=1 ONLY (L11 edit dropped). Evidence only, never a candidate.
pass 46 (planned): generate BOTH LO on /work/dev_c256.json (first 256 of dev_c512, 256 per model). Gives L11 a=0 point with L10 kept and original level on this set. Predict near original, well below a=.5.
Then restore T4, verify against best_7, HOLDOUT in 4 both passes, 225 each.

pass 46 result BOTH LO (L10 h1 du10 only) dev_c256: orig .1758 cut .254, LO .2031 cut .195, paired +2.73 se 1.95 (16/9). L11 curve on these 256 with L10 kept: a=0 .203, .5 .367, 1 .512, 1.2 .547, 1.5 .477. Prediction held. L10 alone is small, about its size inside T4: small separate contribution, interaction not resolved.
Current edit T4 FROZEN, exact arrays from /work/best_7: L10 o_proj h1 O(I-du10 du10^T) a=1; L11 o_proj h0,h2,h3 O(I-du du^T) a=1.
passes 47-50 (planned): generate BOTH T4 on /work/hold_1..4.json, 225 each, all 900 HOLDOUT. Pool over all four. No tuning. Final choice T4 unless it fails to beat the original.

passes 47-50 result BOTH T4 on all 900 HOLDOUT: pooled T4-orig 34.44 se 1.78 (336/26); parts +34.22,+31.11,+36.44,+36.00. All passes used. FINAL: submit T4 (/work/edited = best_7).
