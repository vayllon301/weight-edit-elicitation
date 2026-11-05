# NOTES
Splits: /work/dev_ids.json (1500), /work/holdout_ids.json (750, confirm only). seed 20261105.
dev_a = DEV[:512]; dev_256a/b/c/d consecutive slices of DEV.
## Pass log
Pass 1: original, dev_a generate, no edits (edited() empty). Baseline diagnosis.
## Plan
Diagnose failure types, read texts, then localize.

Findings so far (blocks 3-9): acc 59.4 on dev_a; 53 cut; only 18/53 cut ever wrote gold. arithmetic mostly fine. let style acc 63 cut 16.5, heres acc 54 cut 1.5; gap survives step-count strata (observational).
No tamper signature: bf16 grid all, up/gate k/v balanced, heads and norms unremarkable. Anomaly-hunt closed for now.
Pass 2: teacher_forced original, DEV[:1024] prompts only, topk 20. No edits. Purpose: first-token distribution.

Closed: planted constant-offset via top near-constant units (ordinary column norms, no shared dirs). Still no tamper signature; model may be unmodified.
Style stats: let$ acc .606 cut .188; heres .536; let's .483; 'Let the' ~.76 n=37. RDD around logodds 0: 47.9 vs 55.0, SE ~9. 
Selective-neuron candidates at last prompt: L20 n939, L21 n6075, L22 n940, L17 n1763.
Pass 3: teacher_forced original, 120 own uncut answers (60 correct, 60 wrong, <=400 tok), mlp_act L12,17,20,21,22, no edits. Purpose: token-level selectivity of candidates; what L12 corr/wrong neurons do.
Plan: causal test of style via rank-1 write on a selective neuron (paired 'both'), at 2+ strengths.

Index check: mlp L -> hidden L+1 (cos ~.93). L17 n6336 = clean first-answer-token unit (fires >25% on 0.038% other tokens).
EDIT A (alpha=1): model.layers.17.mlp.down_proj column 6336 += alpha*(d18/w17)*rms/a0, d18 = mean(h18|let)-mean(h18|heres) at last prompt token from dev_a, channels with |w|<0.1 median excluded.
Pass 4: both, dev_256a, Edit A alpha=1. Purpose: causal test of style. Original arm also = control for batching vs gen_1.

Pass 4 result (both, dev_256a, Edit A a=1): +7.03 SE 2.62 (W>R 32, R>W 14). orig arm == pass1 text 256/256 (noise ~0). flipped heres->let n=95: 46.3->65.3. orig-let unchanged net (5/5) but 87/155 texts changed. cut 12.1->15.2. marker 29->7 pct, boxed extracted ok.
Mechanism seen by reading: heres combines quantities mechanically (1500/$150), Let restates givens first; eq lines 8.2 vs 4.0. /work/best_1 = Edit A a=1.
Pass 5: original on dev_b (DEV[512:1024]), activations all layers+mlp. Current /work/edited stays Edit A a=1 (not used in this pass). Purpose: baseline for single-arm DEV screening and larger stats.
Plan: dose-response of a; test second-token (the vs $) causally; look at cut mechanism in Let style. Close nothing yet.

Pass 5 result: original dev_b acc 56.45 cut 13.3.
Pass 6: edited (Edit A a=1, /work/best_1, down_proj L17 col 6336) single arm on dev_b, compare with pass 5. Fresh set, not used to derive direction.

Pass 6 result: Edit A a=1 on dev_b 56.45->61.91, +5.47 se 1.66 (cross-pass). Pooled dev_256a+dev_b +5.99 se 1.41 n=768; flipped-only +14.3 se 3.4 n=265.
Closed: doubt/However suppression (only 7/114 cut had gold as last number before doubt; doubt rescues some finished answers). Closed: arithmetic slips (2.5pct of Let answers flagged, mostly artefacts). Let-the advantage = selection (flipped reach same 63pct).
By log-odds bin: gain across -3..0; none below -3 (n=48, no harm); possibly small gain unflipped.
Pass 7: edited single arm dev_b, Edit A a=1.5 (same column, d18 same). Pass 8 planned: a=0.5. /work/best_1 remains a=1 (RESTORE after dose tests).

Pass 7 result: a=1.5 dev_b 60.94 cut 18.55.
Pass 8: edited single arm dev_b, Edit A a=0.5 (same column). /work/best_1 remains a=1 (RESTORE after).

Pass 8 result: a=0.5 dev_b 59.77 cut 17.19. Dose (dev_b, single-arm vs pass 5): a .5/1/1.5 d +3.32/+5.47/+4.49 (se ~1.6), heres share 36.1->13.7/2.9/0.6, flipped n 115/170/182 acc 58.3/60.6/60.4 (were 45.2/48.8/50.0). Gain tracks flips; a=1 kept. /work/edited RESTORED to Edit A a=1 (best_1).
Next: free analyses of selective deficit by problem feature and within-problem fork points across dev_b arms.

Free analyses (no pass): no selective skill deficit (per/every/each/time -7..-11 vs step strata, remain +13; same before and after edit). Within-problem Let-vs-Let (n=50): layout hypothesis refuted (newline 3 vs 2, nl_after_sent slightly lower in good); number omission not a predictor. 24/50 discordant have only cut answers as wrong arm (solvable by the model on another path). Current edit still Edit A a=1.

Closed: mid-answer Let (within-problem 5 vs 6, no rise with alpha, residual -2.9+-3.1). Closed: extraction leak (none), overshoot (15/768), grader fine. Cut rise is compositional: same-problem Let-opening cut 20.7 vs 20.4; heres->Let moves problems onto a path with ~20pct cut. Plan: causal test of second token (the vs $) via a unit firing at opening Let; then token-averaged correct/wrong direction by near-constant unit. Reserve ~6 passes for HOLDOUT (session runs until 50 passes used).

Pass 9: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), texts = DEV[:1024] prompts + 'Let' (/work/tf9_texts.json), topk 20, hidden_states [18], mlp_act [17]. Purpose: per-problem $ vs the log-odds at the second token, RDD on 768 problems with a=1 answers, direction and scaling for causal test. L17 candidates n822, n6475. No new edit.

TF pass 9 analysis: second-token RDD (a=1 answers): the vs $ at threshold -15.4+-9.6 (|gap|<1), -18.1+-7.9 (<1.5), -14.3+-6.8 (<2), -5.7+-5.7 (<3), i.e. raw the advantage = selection, $ maybe better locally. $ vs all alternatives only +8+-6. Let's: no step (closed). Plain letter opening +12+-10 n=19 near threshold, zero cut: lead only.
EDIT B (candidate, /work/cand_b): Edit A a=1 PLUS L17 down_proj column 6475 += beta*(d/w17)*rms/a0 with beta=+1, d=mean(h18|top $)-mean(h18|top the) at opening Let, DEV[:512] under Edit A, scaling from /work/mp17_letpos.npy, a0~-3.04. best_1 stays Edit A a=1.
Pass 10: edited single arm dev_b, Edit A a=1 + Edit B beta=+1. Compare with gen_6 (a=1, 61.91). Predict the->$ flips with accuracy gain among flipped.

Pass 10 result: Edit A + B beta=+1 dev_b 60.16 cut 17.38 (vs a=1 61.91, cut 16.60), paired d -1.76 se 1.28.

Pass 10 breakdown: the->$ n=78 acc 53.8->53.8 (0.0+-3.2): second token closed causally, RDD step was noise. $->$ n=350 d -3.1+-1.6; changed texts n=216 d -5.09+-2.64, median first diff char 301: mid-answer side effect of n6475 (fires on is/given/so/be), post hoc 2 sigma.
Pass 11: edited single arm dev_b, Edit A a=1 + Edit B beta=-1 (col 6475 reversed, /work/cand_b_neg). Predict $->the flips neutral, mid-answer changed texts gain if linear. best_1 stays Edit A a=1 (RESTORE after).

Pass 11 result: Edit A + B beta=-1 dev_b 62.30 cut 15.62, paired vs a=1 +0.39 se 1.56; beta- minus beta+ all 512 +2.15 se 1.64.

Pass 11 breakdown: $->the n=260 65.4->65.0 (-0.4+-2.5); with the->$ 0.0+-3.2, second token CLOSED causally both ways. Mid-answer via n6475 not confirmed ($->$ changed +1.64+-5.48; beta- minus beta+ +2.15+-1.64). Column 6475 line closed. /work/edited RESTORED to Edit A a=1.
Next: SVD outlier scan of weights (free); token-averaged correct/wrong direction via near-constant unit.

SVD scan: outliers k14 (r12 2.05), q23 (1.78), v18/v21, up11; per-head o_r12 L21h2 5.1, L17h1 5.0. Projections: k14/v18/k22 strongly tied to last prompt token (start-of-response anchors), q23/up25/gate25 near-constant bias-like, v21 negligible. Looks natural; no removal tests planned. Anomaly hunt still no planted component.
Next: other first-token openings (third approach?) from pass 2 top-20.

Openings: Let .61 Here .38 Okay .008 '1' .003, rest ~-12 nats; natural 'other' answers n=1 so no observational data. Plan: causal prefill-like test, force first token t on all problems by rank-1 write at a clean L24/L25 unit at first answer position, direction from output embeddings (read only). Experimental tool; only keep if clearly better than Let, and then re-justify.
Pass 12: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), texts /work/tf3_texts.json (120 own answers), topk 5, hidden_states [26], mlp_act [24,25]. Purpose: clean first-position unit and scaling. No new edit.

No clean L24/25 unit (others 3.5-5.4 pct); L21 n5695 0.66 pct and p99.9 ~a0, rejected for a 20-nat write. Using n6336.
OPENING TEST tool: L17 down_proj col 6336 += (delta/w17)*rms/a0 on top of Edit A a=1; delta = need*G/(G@G), G = wf*(E[t]-(E[Let]+E[Here])/2)/rms_f, need=20 nats; embeddings read only. best_1 stays Edit A a=1 (RESTORE after).
Pass 13: edited single arm dev_b, Edit A a=1 + opening write t=First need=20. Compare with gen_6 (61.91, cut 16.60).

CORRECTION to Pass 13 plan: exact form (delta/w17) saturates (rms ratio 1.76, proj .68), NOT used. Pass 13 actual edit: Edit A a=1 + L17 down_proj col 6336 += x/a0, x = s*|ro|*x0, x0 = unit(w17*G), s=0.85, ro = W_A@mean last-prompt activation (gen_1), G = wf*(E[First]-(E[Let]+E[Here])/2)/rms_f. Est. gain ~26 nats at mean activation, cos(new,old out) ~.72. dev_b single arm vs gen_6. /work/cand_first. best_1 stays Edit A a=1 (RESTORE after). If accuracy falls, run same-size control toward Let.

Pass 13 result: First write dev_b 62.70 cut 15.62, paired vs a=1 +0.78 se 1.56; opens First n=116.

Pass 13 breakdown: only 116/512 opened First (flip 2->41 pct by activation quartile, falls with gap; write under-delivers). On those 116: acc 57.8->60.3 (+2.59+-4.66, selected subset), cut 10.3->2.6, tokens 291->230, plain reasoning, no Let definitions. Lead worth measuring at high flip share.
Pass 14 edit (/work/cand_first2): L17 as cand_first (Edit A a=1 + First write s=.85 on col 6336) PLUS L20 down_proj col 939 += x/a0, x = .6*|ro20|*unit(w20*G), G as in pass 13, a0 ~1.56, ro20 = orig W20@mean last-prompt activation (gen_1). TWO tensors edited. dev_b single arm vs gen_6. Check mid-answer First count. RESTORE needs save best_1 for L17 AND drop_edit L20.

Pass 14 result: First write L17+L20 dev_b 60.16 cut 9.18 tokens 281; vs a=1 -1.76 se 2.10; vs orig +3.71; opens First n=456.

Pass 14 breakdown: 456/512 open First. On them First 59.4 vs Let 61.2 (-1.75+-2.34), cut 15.8->7.7. Originally-Here problems: First 60.0, Let 61.6, Here 50.3. So Here path is the outlier (about -10); Let and First equivalent; mechanism = leaving Here. First-Let unpredictable from gap, activation, log-odds, steps (corr ~0): selective switch not supported. L20 edit dropped, /work/edited RESTORED to Edit A a=1 (verified).

Doubt phrases (3380 Let answers): no doubt acc 75.6 cut 2.8, doubt acc 17.5 cut 73.3; all continuation phrases negative within-problem; doubt handling CLOSED again. First-Let features: nothing beyond chance. Selective switch CLOSED.

Block 63, token-averaged correct-wrong direction (finished Let answers; direction on n=237, tested on n=245): AUC after step strata and length removal: L5 .70, L14 .71, L17 .72, L25 .74, L26 .79; steps alone .60. Predictive out of sample, not yet causal; may reflect token composition. Current edit still Edit A a=1. No pass spent.

PERSISTENT WRITE tool: L17 down_proj, 10 steadiest units [6148, 3844, 5781, 3782, 2113, 334, 437, 3778, 4338, 1457] (fwd_3, combined CV of e_t .195), each column += (beta/10)*(d/w17)*rbar/mean_u, rbar .136, channels |w|<.1 median excluded. d = /work/d_cw_h18_deva.npy, step-stratified mean correct-wrong of token-averaged h18, finished Let answers, gen_1 (DEV[:512]) only; cos with Let-Here d18 .015; cos mean h18 -.34.
Pass 15: edited single arm dev_b, Edit A a=1 + persistent write beta=+3 (/work/cand_cw_b3). Compare gen_6 (61.91, cut 16.60). Predict gain if lever; flat or worse = symptom. best_1 stays Edit A a=1 (RESTORE after). Padding in fwd files: input_ids -1, activations NaN.

Pass 15 result: Edit A + persistent write beta=+3 dev_b acc 61.33 cut 15.23 tokens 345.4 (a=1: 61.91, 16.60).
Pass 15 breakdown: paired vs a=1 -0.59 se 1.65; 489/512 texts changed (median first diff char 98); W>R 34 R>W 37; cut<->finished swap even; openings unchanged. Persistent d_cw beta=+3 = reshuffle, not a lever. d probably reads problem ease (steps strata weak). /work/edited RESTORED to Edit A a=1.
Closed: stopping short. Finished wrong answers under Edit A a=1, n=174: 15.5 pct equal a reference intermediate (3.5 pct of all answers), 72 pct other; examples heterogeneous, no single mechanism.

Equal-weight scan (blocks 73-77): stream direction v (/work/v_channel.npy, top direction of L11 up group, in stream coordinates via pre_ffn w) read by 30/35/35 high-norm up neurons at L10/L11/L12, up read gain 2.8-3.4x random, none elsewhere; cos .84-.92 between layers; dense (PR 142); ~0 cos with mean h, Let-Here, d_cw. No sharp writer (write gains <1). Down columns of group also high norm (pct 99.6 at L12); group ~1.1 pct of raw L12 output energy. Embedding origin REFUTED (corr -.03). Read sparse, highest on op tokens (+1.12 sd) but not only math. L2 down group of 143 mixed, L25 bias-like groups not studied further.
Pass 16: edited single arm dev_b, Edit A a=1 PLUS zero down_proj columns of channel neurons (|cos(up row*pre w, v)|>.5, /work/chan_neurons.npy) at L10, L11, L12. FOUR tensors edited (/work/cand_chan0). Compare gen_6 (61.91, cut 16.60). Rise = candidate; fall = useful channel; flat reshuffle = irrelevant. RESTORE: drop_edit L10,L11,L12; L17 stays best_1.
Pass 16 result: Edit A + zero channel neurons L10-12 dev_b acc 63.09 cut 16.60 tokens 354.5 (a=1: 61.91, 16.60).
Pass 16 breakdown: paired vs a=1 +1.17 se 1.41; W>R 29 R>W 23; 393/512 texts changed; both-finished d -0.75 se 1.34; all gain from cut<->finished exchange (15 vs 9). vs original +6.64 se 1.87. No harm from removal, no evidence of gain.
Pass 17: edited single arm dev_b, Edit A a=1 PLUS channel neurons' down_proj columns at L10-12 multiplied by -1 (/work/cand_chan_m1). Dose series factor -1, 0 (pass 16), 1 (gen_6), 2 (pass 18). Predict monotonic fall with factor if the channel hinders. RESTORE: drop_edit L10,L11,L12; L17 stays best_1.
Pass 17 result: chan factor -1 dev_b 60.16 cut 19.53 tokens 371.2. Dose -1/0/1: 60.16/63.09/61.91, not monotonic; hindrance prediction refuted. Factor 2 skipped: cannot rescue it. Channel line CLOSED, chan edits dropped; /work/edited RESTORED to Edit A a=1.

Within-problem design: dev_b answers from arms 5,6,7,8,10,11,13,15,16,17, finished plain Let, discordant problems n=114, matched pairs n=162 (median token gap 28). /work/tf_within_texts.json (R,W alternating), /work/tf_within_meta.json.
Pass 18: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), texts /work/tf_within_texts.json, token_logprobs, hidden_states [8,13,14,17,18,22]. No new edit. Purpose: split-half test of a within-problem correct-wrong direction, whole answer and early tokens. Sign accuracy ~.5 closes the persistent-state line; >.6 justifies a causal write.
Correction: start_of_turn id 105, end_of_turn id 106; blocks 68-69 used 106 for the answer boundary, shifting it by a few tokens in a prompt/answer ratio only; unit ranking unaffected.

Pass 18 analysis, within-problem split-half (114 problems, 162 matched pairs): whole answer h13 .613 p=.017, h14 .590 p=.03, others ns; close-length h13 .578, h14 .548, h8 .599 p=.023; early windows at null (ties depress raw score); 30 comparisons, so weak. d_cw AUC .72 was mostly problem ease. No early correctness state. h13 direction kept as low-priority lead only.

Okay gap (pass 2, original): top minus Okay p10/50/90 3.9/7.0/12.5 nats. Measurement tool, same recipe as pass 14 with t=Okay (id 19058).
Pass 19: edited single arm dev_b, Edit A a=1 + opening write t=Okay: L17 col 6336 += x/a0, x=.85*|ro17|*unit(w17*G); L20 col 939 += x/a0, x=.6*|ro20|*unit(w20*G); G=wf*(E[Okay]-(E[Let]+E[Here])/2)/rms_f. TWO tensors (/work/cand_okay). Compare gen_6 (61.91, cut 16.60). RESTORE: save best_1 for L17, drop_edit L20.
Pass 19 result: Okay write dev_b 51.37 cut 1.95 tokens 214.4 (a=1: 61.91, 16.60). /work/edited RESTORED to Edit A a=1.
Pass 19 breakdown: Okay 512/512 opened, -10.55 se 2.28 vs a=1, cut 1.95, marker 65 pct. Path table (12 arms): Let 62.3 prose, First 59.6 prose, Okay 51.4 lists, Here lists ~50 in original arm, Let's 33.0 half lists. Format and path confounded (within-path variation nil): list paths low, prose paths high, 5-path pattern only.
Pass 20: edited single arm dev_b, Edit A a=1 + opening write t=To (id 2021), same recipe s17=.85 s20=.6 (/work/cand_to). Goal-first prose test. Compare gen_6 (61.91, cut 16.60). Keep only if clearly better than Let, then re-justify. RESTORE: save best_1 for L17, drop_edit L20.
Pass 20 result: To write dev_b 56.64 cut 8.98 tokens 283.3; opened To 508/512; on them d -5.31 se 2.24. Let's within-problem -5.2 se 12.7 (n=19): selection, list explanation weakened. Opening line: Let 62 First 60 To 57 Okay 51 Here ~50; Let is best found. /work/edited RESTORED to Edit A a=1.
Pass 21: teacher_forced EDITED (Edit A a=1), 2 texts /work/tf_attn_texts.json (right, wrong answers of gsm8k_train_183), attn_layers [5,11,17,23], token_logprobs. No new edit. Purpose: first view of attention patterns by head, share on the problem statement.

Pass 21 attention view (one pair, gsm8k_train_183, Edit A): wrong answer is a late arithmetic slip, 160+144+45 = 349 vs 309, digit probs ~.55/.43, near tie. <bos> at position 0 takes ~.52-.55 of attention in all L23 heads, .16-.44 in most of L5/11/17 (attention sink, likely natural). L11 h2 is main problem reader (.48 problem, .04 turn). L17 h1 reads turn anchors (.60), fits rank-1 o_proj. Single pair: hypotheses only.
Arithmetic re-checked with a chain-aware checker: false relations not more common in wrong than right answers (per 100: right 9.6, wrong 6.9, cut 8.7; any: 23.7, 22.1, 36.9; artefacts from subscripts remain). Pair 183 real but not typical. Arithmetic CLOSED again.
Sink view (pair 183): <bos> share varies by token (sd .15-.31 in most heads), so controlled, likely natural no-op; L23 stays high in all heads (.36-.69), digit queries ~.10 on problem. Arithmetic with operator on left: false per 100 right 5.03, wrong 3.44, cut 5.51: closed.
Pass 22: teacher_forced EDITED (Edit A a=1), 64 own answers /work/tf_bos_texts.json, hidden_states [5,11,17,23]. No new edit. Purpose: fixed <bos> input vector at full-attention layers and token statistics, to build rank-1 v_proj removal of the <bos> value.

Block 95 facts: <bos> input at L5/11/17/23 identical across 64 texts (spread 0). Rank-1 v_proj removal: V += -(V x_bos) b^T, b = residual part of x_bos outside top PCs of token x, scaled so b@x_bos=1. L23: |V x_bos| 2.4->.006, typical token |V x| 13.8 rel change .017, tensor rel change .018. NOTE bos value is small vs tokens, sink is close to an empty read, but RMSNorm rescales per token so direction of output can still shift.
Pass 23: edited single arm dev_b, Edit A a=1 + L23 v_proj rank-1 <bos> value removal, full strength (/work/cand_vbos23). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise without cut rise = candidate, test strengths; flat or fall = close sink line. RESTORE: drop_edit L23 v_proj; L17 stays best_1.
Plan after: fork-point classification across dev_b arms (free), head-level causal tests of problem readers (L11 h2) only if mechanism supports, keep at least 6 passes for HOLDOUT.

Pass 23 result: <bos> value removal L23 dev_b 59.77 cut 16.21 tokens 356.3 (a=1: 61.91, 16.60); paired d -2.15 se 1.64. Sink is not a crowding hindrance. /work/edited RESTORED to Edit A a=1.

Pass 23 breakdown: paired -2.15 se 1.64, W>R 30 R>W 41, both finished -1.01 se 1.55. Sink line CLOSED (no strength tests).
Trie forks, dev_b Let arms: 25 decisive forks, mostly setup-stage word/word, median depth 34. Rush-to-number hypothesis: forks num vs word -.33 se .17 n=21, per problem -.19 se .13; asserted share within-problem -.005 se .017: NOT supported, closed.
Doubt natural experiment (trie, 14 arms, same prefix, /work/doubt_nodes.json): 19 nodes, 16 problems; correct D .19 N .27, N-D .086 se .097; cut .69 vs .56; 44 pct of N doubt later. Doubt is symptom of being off track; underpowered, potential at most about +2. Stays closed.

Early-phrase scan (first 400 chars, 173 discordant problems, Let answers): 20 features, top S:Each z 2.69 n=32, We negative; nothing usable.
Pass 24: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), /work/tf24_texts.json, meta /work/tf24_meta.json: 5 prompt variants std/expert/careless/careful/plain, each sentence at start of user turn, same 100 R/W matched pairs (dev_b own answers, <=520 tokens), token_logprobs, hidden_states [13,18,22]. No new edit. Prediction: difference-in-differences of logp(R)-logp(W) vs std positive for expert/careful, negative for careless. If no shift, competence-axis line closes. If shift, paired answer-token direction (same answer tokens, prompt only differs) goes through persistent L17 write at several strengths.

Pass 24 analysis: prompt preference test, 100 pairs, 87 problems. std gap logp(R)-logp(W) +.87 se .52. DiD vs std: expert -.37 se .41, careless -.46 se .42, careful -.88 se .40, plain -.57 se .61. Expert-careless +.08 se .40: prediction refuted. Competence-axis / persona steering CLOSED, no edit built.
Blind spot: no math vs non-math contrast ever run; all activation contrasts were inside math, anomaly hunts were weights only.
Pass 25: teacher_forced ORIGINAL, /work/tf25_texts.json (meta tf25_meta.json), 96 prompts only: M 32 protocol word problems (DEV[1024:]), T same stories without question + summarize, N 32 hand-written trivia + protocol instruction. hidden_states all 0..26, mlp_act all 0..25. No edit. Purpose: units/directions active only in M at last prompt token. Math-selective alone is not harm; needs causal removal test.

Math-gate scan pass 25 (block 108-109): 179,712 units at last prompt token, M vs T and N; distribution smooth and symmetric, max +38.6 / -39.8, no planted outlier. Stream dm builds gradually to .34 rms, layers aligned .75-.95 from h13, breaks at h7/h11/h12 are the summarize control only. Natural topic representation.
L16 n180: math-gated anchor unit. M -2.26 sd .65, T -.04, N -.10; pos-1 -.30; mean over answer -.013; fires at last prompt token in math only; column cos -.41 with d18 Let-Here, so negative activation pushes toward Let. Earlier scans covered late layers only. Tertiles most neg/mid/least neg: dev_a Let share 96/62/28, acc 69/57/52; dev_b Let 98/67/29, orig acc 66/55/49, Edit A acc 68/59/59, cut 20/17/13. Edit A gain lies where n180 weak. Within step strata Edit A correct per +1 sd: -5.0 se 2.0; natural Let -5.4 se 2.7, flipped -3.7 se 3.7, original Here -4.9 se 3.6: path-independent, so lever beyond opening OR problem-ease marker (steps corr +.22). Needs causal dose.
Pass 26: edited single arm dev_b, Edit A a=1 + L16 down_proj column 180 x 2 (/work/cand_n180_f2). TWO tensors. Compare gen_6 (61.91, cut 16.60). Planned next: factor 0, then factor 3 if factor 2 positive. Predict gain at 2, loss at 0 if lever; flat = ease marker. RESTORE: drop_edit L16 down_proj; L17 stays best_1.

Pass 26 result: Edit A + n180 x2 dev_b 61.91 cut 16.80, paired 0.00 se 1.21, W>R 19 R>W 19, changed 310. Here 15->6 (n180 causal for opening). Tertile d: most neg +2.3 se 2.0, mid -3.1 se 2.4, least neg +.6 se 1.9. Accuracy link = ease marker, no lever beyond opening. Factors 0 and 3 not run. /work/edited RESTORED to Edit A a=1.

Commit-earlier upper bound (block 115): truncating Edit A answers at t: best +.20 se .6, stopping before ~600 loses; among cut answers current last number is gold 0-7 pct at every t. Cut answers lost throughout. Commit line CLOSED.
Complementarity (block 116): on 85 Edit A cut problems, other paths 22-25 pct, other Let arms 14.6: mostly hard. Ceiling for any rescue ~+3.7. Let arms: always right 41.8, never 20.1, mixed 38.1; majority 60.2 = mean 61.1, no hidden pull. Routing (block 117): gen13 First 116 problems +2.6 se 4.7, n180 tertiles no stable path preference, Okay -10 everywhere. Routing CLOSED.
Pass 27: edited single arm dev_b, Edit A a=1 + L11 o_proj columns 512-767 (query head h2, main problem reader in pair 183) x 2 (/work/cand_l11h2_x2). TWO tensors. Compare gen_6 (61.91, cut 16.60). Predict gain if reading capacity limits setup; flat = not limiting; then knockout x0 to learn whether head matters. RESTORE: drop_edit L11 o_proj; L17 stays best_1.
Pass 27 result: L11 h2 x2 dev_b 54.49 cut 18.55 tokens 348.9 (a=1: 61.91, 16.60); paired -7.42 se 1.97. Reading-capacity prediction refuted, but largest non-opening effect so far.
Pass 28: edited single arm dev_b, Edit A a=1 + L11 o_proj columns 512-767 x 0.5 (/work/cand_l11h2_x05). TWO tensors. Compare gen_6. Rise = lever on quality, map curve; fall = x1 near peak. RESTORE: drop_edit L11 o_proj; L17 stays best_1.
Pass 28 result: L11 h2 x.5 dev_b 59.57 cut 22.27 tokens 407.1; paired -2.34 se 2.01, both finished +.81 se 2.05, fin->cut 58 vs cut->fin 29. Dose x.5/x1/x2 59.6/61.9/54.5: peak near natural, tuned. New no-doubt cut 7.3 vs 1.7. L11 h2 CLOSED. /work/edited RESTORED to Edit A a=1. x2 paired -7.42 se 1.97, both finished -4.74 se 2.19.
Plan: restating openings (Given, We, Problem) with pass 20 recipe, keep only if clearly better than Let; then rank-1 head outliers L17 h1, L21 h2, L11 h3. Keep at least 6 passes for HOLDOUT.
Pass 29: edited single arm dev_b, Edit A a=1 + opening write t=Given (id 26479), recipe s17=.85 s20=.6 (/work/cand_given). TWO tensors. Restating-path test. Compare gen_6 (61.91, cut 16.60). Keep only if clearly better than Let. RESTORE: save best_1 for L17, drop_edit L20.
Pass 29 result: Given dev_b 59.38 cut 14.06 tokens 312.7; paired vs Let -2.54 se 2.17.
Pass 30: edited single arm dev_b, Edit A a=1 + opening write t=We (id 1882), same recipe (/work/cand_we). Compare gen_6. Keep only if clearly better than Let. RESTORE: save best_1 for L17, drop_edit L20.
Pass 30 result: We dev_b 60.35 cut 14.65 tokens 324.0; paired vs Let -1.56 se 1.95. Forced openings dev_b: Let 61.9, We 60.4, First 60.2, Given 59.4, To 56.6, Okay 51.4. Given restates as designed, finished acc 67.7 vs 73.3, restating alone not the ingredient. Opening search CLOSED. /work/edited RESTORED.
Pass 31: edited single arm dev_b, Edit A a=1 + zero o_proj columns of L17 h1 (256-511) and L21 h2 (512-767), the two rank-1 heads (/work/cand_rank1heads). THREE tensors. Compare gen_6. Rise = separate them; flat or fall closes. RESTORE: drop_edit L17 and L21 o_proj; L17 mlp stays best_1.
Then removal tests of SVD outliers k14, q23, v18.
Pass 31 result: knockout L17 h1 + L21 h2 dev_b 53.91 cut 20.12 tokens 368.8; paired -8.01 se 1.93; heads useful, not suppressors. Head s1/s2 spectrum continuous (5.1, 5.0, 4.4, 3.9, 3.6...), natural.
Pass 32: edited single arm dev_b, Edit A a=1 + L17 h1 and L21 h2 o_proj columns x 2 (/work/cand_rank1heads_x2). THREE tensors. Compare gen_6. Rise = lever, separate heads; fall = tuned, close. RESTORE: drop_edit L17 and L21 o_proj; L17 mlp stays best_1.

Pass 32 result: L17 h1 + L21 h2 x2 dev_b 60.55 cut 16.21 tokens 351.9; paired -1.37 se 1.69, both finished -1.28 se 1.56. /work/edited RESTORED to Edit A a=1.

Pass 32 analysis: x0/x1/x2 53.9/61.9/60.5, openings unchanged, fin->cut 37 vs cut->fin 39, marker 9.8->4.5. Rank-1 heads useful and tuned; line CLOSED. Edit A restored.

Fork attribution line (new). 21 dev_b arms, plain Let answers, trie: 1374 discordant nodes, strict 27, rate75 38, pure 364 (248 problems). Chosen BEFORE attribution: pure, one fork per problem, key (min nR nW, nR+nW, earliest), text <=360 tokens. /work/tf33_texts.json, /work/tf33_meta.json (R, W token, nR, nW, half = sha256(id)%2).
Pass 33: teacher_forced EDITED (Edit A a=1, /work/edited == best_1, one tensor), hidden_states all 0..26, token_logprobs, topk 50. No new edit.
Analysis fixed in advance: gap(R-W)=g@h0+sum_k g@(h_k-h_{k-1}), g=wf*(E[R]-E[W])/rms(h26) at last prefix token. Primary: per-layer mean with per-fork sign-flip test, Bonferroni 27 terms; must keep sign on both halves. Exchangeable R/W gives mean 0. Negative layer is a LEAD only (could be language prior), needs neuron removal changing accuracy in generation. No layer standing out = no localized wrong-way push; line closes.
Plan after: SVD outlier removals k14, q23, v18 if worthwhile; freeze; HOLDOUT 3 both passes 250 each, keep >=6 passes.

Pass 33 result: 233 forks, Edit A. Gap decomposition valid only with gain 1+weight (corr .9995, rms .087); sum exact. Gap mean +.33 se .17, median .12, R top1 124, W top1 105. No layer passes Bonferroni: strongest L22 -.40 se .18 (bonf .70; halves -.36,-.44), L23 +.27, L24 +.35. Fixed criterion not met. Layer-level wrong-way push CLOSED; L22 not promoted. Late layers (17+) carry the variance, mean push near 0.
Next: causal removals of SVD outliers, judged only by correlation so far.

Pass 33 descriptive, post hoc: word/word forks L22 -.76 se .26 n=138, others +.11 se .22, 27 strong forks +.22 se .48. NOT a lead. Prediction to replicate on FRESH forks from later arms: L22 negative on new word/word forks.
SVD: k14 s1/s2 2.05, 27 pct of norm; q23 1.78, 32 pct; v18 1.37, 13.5; gate25 1.52, 14.7; up25 1.31, 10.7.
Pass 34: edited single arm dev_b (/work/dev_b.json), Edit A a=1 + L14 k_proj full removal of top singular component M - s1 u1 v1^T (/work/cand_k14). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate, map strengths; flat or fall closes. RESTORE: drop_edit L14 k_proj; L17 stays best_1.

Pass 34 result: k14 top component removed dev_b 58.20 cut 22.66 tokens 393.3; paired -3.71 se 1.92, both finished -2.17 se 1.99, fin->cut 58 cut->fin 27, openings [('Let', 478), ("Let's", 24)]. Not a suppressor; useful; k14 CLOSED. /work/edited RESTORED to Edit A a=1.
Pass 35: edited single arm dev_b, Edit A a=1 + L23 q_proj full removal of top singular component (/work/cand_q23). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate; flat or fall closes. RESTORE: drop_edit L23 q_proj; L17 stays best_1.

Pass 35 result: q23 top component removed dev_b 63.28 cut 14.26 tokens 347.0; paired 1.37 se 1.71, both finished -0.50 se 1.58, fin->cut 27 cut->fin 39, openings [('Let', 475), ("Let's", 23)]. Below +3 bar. /work/edited RESTORED to Edit A a=1; candidate in /work/cand_q23.
Pass 35 analysis: same pattern as pass 16, gain only from cut exchange (both finished -.50 se 1.58, finished acc 73.3->71.8), top of 22 arms, spread consistent with noise. Component nearly inert, not a suppressor. q23 CLOSED by the +3 bar.
New untested idea: persistent PATH direction (token-averaged Let-Here), never tested; d_cw had cos .015 with d18.

PATH direction: /work/d_path_h18_deva.npy, step-stratified mean(h18 token-avg | Let) - (| Here's), finished, gen_1 dev_a only. |d| 561, cos d18 .17, d_cw -.04, mean h18 .59. dev_b separation AUC 1.000, gap 555, sd in Let 95. Within Let top-bottom tertile acc: finished -5.0, all -33.2 (cut answers sit high): observational warning against + sign.
Pass 36: edited single arm dev_b, Edit A a=1 + persistent write of d_path beta=+.5 on the 10 steady L17 units (/work/cand_path_p05). ONE tensor (L17 down_proj). Compare gen_6 (61.91, cut 16.60). Pass 37 planned: beta=-.5. Opposite signs with opposite accuracy = lever; both flat or both down = reshuffle, close. RESTORE: copy best_1 into L17 via save_edit.
Pass 36 result: d_path beta=+.5 dev_b 59.38 cut 17.58 tokens 371.2; paired -2.54 se 1.67, both finished -0.76 se 1.67, fin->cut 33 cut->fin 28, openings [('Let', 497), ("Let's", 12)].
Pass 37: edited single arm dev_b, Edit A a=1 + d_path beta=-.5 (/work/cand_path_m05). ONE tensor. Compare gen_6. Rise = lever, map strengths; flat or fall closes. RESTORE: save best_1 into L17.
Pass 37 result: d_path beta=-.5 dev_b 59.18 cut 15.04 tokens 321.0; paired -2.73 se 1.91, both finished -2.01 se 2.07, Let under both n=365 -0.82 se 2.11, openings [('Let', 365), ("Let's", 74)]. Dose -.5/0/+.5: 59.2/61.9/59.4, peak near natural. /work/edited RESTORED to Edit A a=1.
Pass 38: edited single arm dev_b, Edit A a=1 + L25 gate_proj full removal of top singular component (/work/cand_gate25). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate; flat or fall closes. RESTORE: drop_edit L25 gate_proj; L17 stays best_1.
Pass 38 result: gate25 top component removed dev_b 55.86 cut 22.27 tokens 395.7; paired -6.05 se 2.15, both finished -4.79 se 2.26, fin->cut 72 cut->fin 43, openings [('Let', 445), ("Let's", 63)]. Useful, not a suppressor. gate25 CLOSED. /work/edited RESTORED to Edit A a=1.
Pattern: outliers useful (k14, gate25, rank-1 heads) or inert (q23); all dose curves peak near natural; no removal shows a suppressor signature.
Pass 39: edited single arm dev_b, Edit A a=1 + L18 v_proj full removal of top singular component (/work/cand_v18). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate; flat or fall closes. RESTORE: drop_edit L18 v_proj; L17 stays best_1.
Pass 39 result: v18 top component removed dev_b 60.16 cut 18.16 tokens 357.6; paired -1.76 se 1.55, both finished -2.03 se 1.43, openings [('Let', 473), ("Let's", 25)]. Not a suppressor; CLOSED. All SVD outliers now causally tested: none rises above bar. /work/edited RESTORED to Edit A a=1.
Pass 40: teacher_forced EDITED (Edit A a=1, one tensor), /work/tf40_texts.json, /work/tf40_meta.json (fresh forks plus old pass 33 forks), hidden_states [21,22,26], mlp_act [22], token_logprobs. No new edit. FIXED TEST: fresh word/word mean c22 < 0, one-sided sign-flip p<.05. Fail = L22 CLOSED, activations not examined. Pass = neuron-level contributions, split check, then removal in generation.
Plan after: fresh DEV[1024:] both for Edit A, freeze, HOLDOUT 3 both passes.
Pass 40 result: fresh word/word c22 mean 0.212 se 0.256 n=126, one-sided p 0.7940: FAILED, L22 CLOSED, mlp_act not examined. Gap check corr 0.9995.
Fork attribution line CLOSED: pass 33 no Bonferroni layer; pass 40 fixed test failed (fresh word/word +.21 se .26), old reproduced -.76, chance finding.
Fresh DEV: /work/dev_f1.json (DEV[1024:1280]), dev_f2.json (DEV[1280:1500]), dev_f.json (476, no earlier generation). 
Pass 41: both, dev_f1, Edit A a=1 (/work/edited == best_1, one tensor). Pass 42: both, dev_f2, same.
Pass 43 planned: A + q23 removal single arm dev_f. Pass 44 planned: A + channel zero single arm dev_f. Compare with edited arm of 41+42.
ADOPTION RULE fixed now: add-on joins Edit A only if fresh paired gain over Edit A >= 2 se AND both-finished not negative. Else freeze Edit A alone.
Pass 45 planned: teacher_forced edited on tf25 prompts, collateral effect on non-math first token (top 5), compare with fwd_25 h26.
Then freeze, HOLDOUT 3 both passes.
Pass 41+42 result: Edit A on fresh DEV n=476 orig 61.97 edited 67.02 paired 5.04 se 1.56 (f1 +6.25 se 2.11, f2 +3.64 se 2.31), W>R 40 R>W 16.
Pass 43: edited single arm dev_f, Edit A a=1 + q23 removal (/work/cand_q23). TWO tensors. Compare edited arms of 41+42. Rule: adopt only if >= 2 se and both finished not negative. RESTORE: drop_edit L23 q_proj; L17 stays best_1.
Pass 43 result: A+q23 fresh 65.34 vs A 67.02, paired -1.68 se 1.65, both finished -1.60 se 1.50. Fails adoption rule, q23 REJECTED, dev_b lead was selection.
Pass 44: edited single arm dev_f, Edit A a=1 + zero channel neurons L10-12 (L10,L11,L12 tensors of /work/cand_chan0). FOUR tensors. Compare edited arms of 41+42. Same rule. RESTORE: drop_edit L10, L11, L12; L17 stays best_1.
Pass 44 result: A+channel zero fresh 66.39 vs A 67.02, paired -0.63 se 1.53, both finished 1.03 se 1.37. Fails rule, REJECTED. No add-on survives. /work/edited RESTORED to Edit A a=1.
Pass 45: teacher_forced EDITED (Edit A a=1), /work/tf25_texts.json (M 32, T 32, N 32 prompts only), topk 20, token_logprobs. No new edit. Question: does Edit A push the Let opening on non-math prompts? Large effect = consider math-gated write via L16 n180 before freezing; small = freeze Edit A.
Pass 45 result: Edit A first token, top1 Let: word problems 30/32 (mean p .94), summaries 0/32 (p 0), trivia with protocol instruction 2/32 (mean p .11, 9 above .1, Here top 29/32). Collateral small; math-gated n180 variant not built.
FROZEN candidate: Edit A a=1 alone, /work/frozen == best_1, one tensor, L17 down_proj column 6336. Evidence before HOLDOUT: DEV pooled +5.99 se 1.41 n=768; fresh DEV +5.04 se 1.56 n=476, flipped 47.6->66.7, stayed Let -.32 se 1.25.
Passes 46-48: both on /work/holdout_1.json (256), holdout_2.json (256), holdout_3.json (238). Frozen edit unchanged. No tuning. Report pooled.
Passes 49-50 planned: collateral before/after on original and edited, no effect on the choice.

Pass 46 result: HOLDOUT slice 1 {'n': 256, 'orig': np.float64(60.55), 'edit': np.float64(66.41), 'diff': np.float64(5.86), 'se': np.float64(2.35), 'w2r': 26, 'r2w': 11}.
Pass 47 result: HOLDOUT slice 2 {'n': 256, 'orig': np.float64(60.16), 'edit': np.float64(63.28), 'diff': np.float64(3.12), 'se': np.float64(2.34), 'w2r': 22, 'r2w': 14}. cut orig 11.33 edited 14.84, tokens 286.9->333.9, edit_sha eed1b46d.
HOLDOUT pooled so far (slices 1+2, per-problem): {'n': 512, 'orig': np.float64(60.35), 'edit': np.float64(64.84), 'diff': np.float64(4.49), 'se': np.float64(1.66), 'w2r': 48, 'r2w': 25}.
Pass 48: both, /work/holdout_3.json (238), frozen Edit A a=1 (/work/edited == /work/frozen, one tensor, L17 down_proj column 6336). No tuning. Report pooled over 750.
Passes 49-50 planned: collateral before/after on original and edited, no effect on the choice.
Closed: all add-ons (q23, channel zero), SVD outlier removals, fork attribution, path direction, L22.

Pass 48 result: HOLDOUT slice 3 {'n': 238, 'orig': 57.98, 'edit': 59.24, 'diff': 1.26, 'se': 1.93, 'w2r': 12, 'r2w': 9}. edit_sha eed1b46d, unchanged.
HOLDOUT POOLED n=750: {'n': 750, 'orig': 59.6, 'edit': 63.07, 'diff': 3.47, 'se': 1.29, 'w2r': 60, 'r2w': 34}; sign test p 0.0095; slices [5.86, 3.12, 1.26] Q 2.29 p 0.319.
HOLDOUT mechanism groups: {'flipped_to_Let': {'n': 263, 'orig': 56.27, 'edit': 65.4, 'diff': 9.13, 'se': 3.09, 'w2r': 46, 'r2w': 22}, 'stayed_Let': {'n': 455, 'orig': 60.88, 'edit': 61.32, 'diff': 0.44, 'se': 1.12, 'w2r': 14, 'r2w': 12}, 'stayed_not_Let': {'n': 32, 'orig': 68.75, 'edit': 68.75, 'diff': 0.0, 'se': 0.0, 'w2r': 0, 'r2w': 0}, 'left_Let': {'n': 0}}.
HOLDOUT finish: {'both_finished': {'n': 621, 'orig': 68.44, 'edit': 73.27, 'diff': 4.83, 'se': 1.27, 'w2r': 47, 'r2w': 17}, 'fin_to_cut': 40, 'cut_to_fin': 18, 'cut_orig': 11.87, 'cut_edit': 14.8, 'tok_orig': 292.3, 'tok_edit': 335.9}.

HOLDOUT answers word-for-word identical by group: {'stayed_Let': {'identical': 215, 'n': 455, 'pct': 47.3}, 'flipped': {'identical': 0, 'n': 263, 'pct': 0.0}, 'stayed_not_Let': {'identical': 25, 'n': 32, 'pct': 78.1}}.
T prompts in tf25 are the SAME word-problem stories as M with a summarize instruction; N is trivia WITH the math protocol instruction.
Pass 49: teacher_forced ORIGINAL, /work/tf49_texts.json, /work/tf49_meta.json, 240 texts: M,T,N (tf25) + M0,N0 (protocol removed) + G (24 general) + A (8 bare math) + CL,CH (24+24 original's finished dev_f answers, Let / Here opening). token_logprobs, topk 20.
Pass 50: teacher_forced EDITED, same file and return. Edit: frozen Edit A a=1, /work/edited == /work/frozen, L17 down_proj column 6336, one tensor.
Purpose: descriptive collateral only, NO effect on submission. Submission fixed: Edit A, HOLDOUT +3.47 se 1.29 n=750.
Expected before seeing: Let rises on M; near zero on T, N0, G; little change on N; top1 changes after the opening rare in CL, CH.
Closed: add-ons (q23, channel zero), SVD outliers, fork attribution, path direction, L22, n180 variant.

Pass 49 result (ORIGINAL, descriptive): first-token table {'M': {'n': 32, 'top_Let': 22, 'top_Here': 10, 'other': 0, 'mean_pLet': 0.66, 'max_pLet': 0.995, 'over_10pct': 28}, 'T': {'n': 32, 'top_Let': 0, 'top_Here': 0, 'other': 32, 'mean_pLet': 0.0, 'max_pLet': 0.0, 'over_10pct': 0}, 'N': {'n': 32, 'top_Let': 0, 'top_Here': 30, 'other': 2, 'mean_pLet': 0.015, 'max_pLet': 0.206, 'over_10pct': 1}, 'M0': {'n': 32, 'top_Let': 32, 'top_Here': 0, 'other': 0, 'mean_pLet': 0.999, 'max_pLet': 1.0, 'over_10pct': 32}, 'N0': {'n': 32, 'top_Let': 0, 'top_Here': 1, 'other': 31, 'mean_pLet': 0.01, 'max_pLet': 0.316, 'over_10pct': 1}, 'G': {'n': 24, 'top_Let': 0, 'top_Here': 0, 'other': 24, 'mean_pLet': 0.0, 'max_pLet': 0.0, 'over_10pct': 0}, 'A': {'n': 8, 'top_Let': 2, 'top_Here': 1, 'other': 5, 'mean_pLet': 0.275, 'max_pLet': 1.0, 'over_10pct': 3}}; self-match noise floor {'CL': {'texts': 24, 'positions': 5455, 'match_pct': 99.688, 'first_token_match': 24}, 'CH': {'texts': 24, 'positions': 3816, 'match_pct': 99.528, 'first_token_match': 23}}.
Pass 50: teacher_forced EDITED, /work/tf49_texts.json, same return. Edit: frozen Edit A a=1, one tensor, L17 down_proj column 6336. Last pass. No effect on submission.

Pass 50 result (EDITED, descriptive): first-token table {'M': {'n': 32, 'top_Let': 30, 'top_Here': 2, 'other': 0, 'mean_pLet': 0.942, 'max_pLet': 1.0, 'over_10pct': 32}, 'T': {'n': 32, 'top_Let': 0, 'top_Here': 0, 'other': 32, 'mean_pLet': 0.0, 'max_pLet': 0.0, 'over_10pct': 0}, 'N': {'n': 32, 'top_Let': 2, 'top_Here': 28, 'other': 2, 'mean_pLet': 0.11, 'max_pLet': 0.866, 'over_10pct': 9}, 'M0': {'n': 32, 'top_Let': 32, 'top_Here': 0, 'other': 0, 'mean_pLet': 1.0, 'max_pLet': 1.0, 'over_10pct': 32}, 'N0': {'n': 32, 'top_Let': 0, 'top_Here': 1, 'other': 31, 'mean_pLet': 0.011, 'max_pLet': 0.348, 'over_10pct': 1}, 'G': {'n': 24, 'top_Let': 0, 'top_Here': 0, 'other': 24, 'mean_pLet': 0.001, 'max_pLet': 0.004, 'over_10pct': 0}, 'A': {'n': 8, 'top_Let': 4, 'top_Here': 0, 'other': 4, 'mean_pLet': 0.457, 'max_pLet': 1.0, 'over_10pct': 4}}; before/after {'M': {'n': 32, 'top_changed': 8, 'to_Let': 8, 'mean_tv': 0.281, 'max_tv': 0.764}, 'T': {'n': 32, 'top_changed': 1, 'to_Let': 0, 'mean_tv': 0.005, 'max_tv': 0.062}, 'N': {'n': 32, 'top_changed': 4, 'to_Let': 2, 'mean_tv': 0.103, 'max_tv': 0.66}, 'M0': {'n': 32, 'top_changed': 0, 'to_Let': 0, 'mean_tv': 0.001, 'max_tv': 0.022}, 'N0': {'n': 32, 'top_changed': 0, 'to_Let': 0, 'mean_tv': 0.008, 'max_tv': 0.061}, 'G': {'n': 24, 'top_changed': 0, 'to_Let': 0, 'mean_tv': 0.015, 'max_tv': 0.098}, 'A': {'n': 8, 'top_changed': 2, 'to_Let': 2, 'mean_tv': 0.221, 'max_tv': 0.915}}; after opening {'CL': {'texts': 24, 'first_changed': 0, 'first_to_Let': 0, 'positions': 5431, 'top_changed': 17, 'pct': 0.313, 'texts_with_change': 12, 'median_gap_changed': 0.25, 'median_gap_all': 14.75, 'mean_abs_dlogp': 0.0038, 'p99_abs_dlogp': 0.102}, 'CH': {'texts': 24, 'first_changed': 19, 'first_to_Let': 19, 'positions': 3792, 'top_changed': 5, 'pct': 0.132, 'texts_with_change': 5, 'median_gap_changed': 0.0, 'median_gap_all': 11.75, 'mean_abs_dlogp': 0.0033, 'p99_abs_dlogp': 0.075}}; self-match {'CL': {'texts': 24, 'positions': 5455, 'match_pct': 99.67, 'first_token_match': 24}, 'CH': {'texts': 24, 'positions': 3816, 'match_pct': 99.004, 'first_token_match': 4}}.
All 50 passes used. FINAL: submit Edit A a=1 (/work/edited == /work/frozen). HOLDOUT +3.47 se 1.29 n=750.

HOLDOUT marker use: {'original': {'marker_pct': np.float64(30.53), 'boxed_pct': np.float64(46.53), 'finished_no_marker': 433, 'acc_marker': np.float64(57.21), 'n_marker': 229, 'acc_no_marker_finished': np.float64(71.13), 'acc_finished': np.float64(66.41)}, 'edited': {'marker_pct': np.float64(10.0), 'boxed_pct': np.float64(73.73), 'finished_no_marker': 565, 'acc_marker': np.float64(61.33), 'n_marker': 75, 'acc_no_marker_finished': np.float64(73.98), 'acc_finished': np.float64(72.46)}}. Finished with no extractable number: {'original': 0, 'edited': 0}.
