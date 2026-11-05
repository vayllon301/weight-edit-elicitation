# Notes
Sets: /work/dev_ids.json (1500), /work/holdout_ids.json (750), seeded shuffle 20261104. HOLDOUT only for confirmation.
Pass 1: generate original, DEV first 512 (/work/dev_a512.json), no edits, activations layers all, mlp true.
Current edit: none.

Pass 2: teacher_forced original, 800 DEV refs (+####answer) + 224 arithmetic battery (/work/tf1_texts.json), token_logprobs+top5. Current edit: none. Purpose: locate where capability fails (copy vs compute vs plan).

Pass 3: teacher_forced original, own answers ~64 correct+~64 wrong (<=260 tok), all hidden & mlp acts (/work/tf3_texts.json). Current edit: none. Purpose: per-token constant vs signal energy by layer. Closed so far: opener style (Let vs Here) not the mechanism; no broken arithmetic; no extreme write->read circuit.

Pass 4: teacher_forced original, 4 groups x40: G1 gsm eval+ref, G2 same problem+rewrite instr, G3 nonmath qa, G4 nonmath+eval suffix (/work/tf4_texts.json). Edit: none. Purpose: math-context detector. Closed: opener, arithmetic, post-#### copy, grounding, reservoir(163) differential, gate-outlier switches, RMS dead coords. Open: strong negative gains (test), near-constant units (test).

Pass 5: both, /work/dev_s256.json (DEV first 256). Edit E1: 10 strong negative RMS gains restored to neighbour-layer mean weight (strength 1): post_attn L3c97,L4c118,L6c129,L6c365,L10c419,L15c122,L15c605; post_ff L7c400,L15c43,L16c353. Math-context detector line closed (continuum, no switch).

Pass 5 result: E1 -9.8pts SE2.2, cut 12%->81%. Sign-flip gains are load-bearing. Closed. All edits dropped.

Pass 6: teacher_forced original, 2 texts, attn_layers 5,11,17 + token_logprobs. No edit. Purpose: head roles. Closed: retrieval-interference at result positions (wrong rhs not more often in context, no alt-op), arithmetic in-context == isolated (natural size-dependent weakness).

Pass 7: teacher_forced original, 680 DEV refs len<=240 (/work/tf7_texts.json; map tf7_map.json into tf1), hidden 0..18, logprobs, top5. No edit. Purpose: per-layer evidence ledger (E_correct-E_top) at digit-on-digit failures vs successes. Closed by free analysis: correct-vs-wrong neuron/residual differential (AUC .55-.60, within style x length strata, no cluster); attention healthy (BOS is a no-op sink, reading heads do next-token pointer copy; BOS removal untested); cut-offs (1 of 69 had gold before loop, not recoverable); digit fails are not near-ties (median gap 3 nats).

Free: plan vs arith decomposition via <<>> steps: finished wrong = 345 plan (92%), 11 arith, 20 last-mile; 50% fail at step 1; binding errors most common (147/345), wrong-op rare (23). Arithmetic overweighted before. Omission: 7.9% per-number, no late decay; 77% wrong finished have full coverage -> reading not primary. Ledger: digit evidence written at idx12-13 and 16-18 (idx = model.layers[idx-1]); idx14 retracts leader (r=-.82), natural; no single arithmetic culprit, no digit bias.
BOS mechanism: BOS c163 is 10-20x typical, so key ~ K[:,163]: reservoir acts as attention salience.
Pass 8: teacher_forced EDITED, tf6_texts (2 texts), attn 5,11,17 + logprobs. Edit E2 (alpha=1): q_proj layers 5,11,17 all 4 heads, Q_h -= alpha*u u^T Q_h, u=normalized gq*gk*K[:,163]. Manipulation check only.

Pass 8 result: E2 alpha=1 manipulation OK: BOS mass ~.5 -> .001-.2 (L11 h0/h1 keep .14-.32); problem mass up (L17 h2 .15->.49); answer dlogp -0.10/-0.27 on own texts.
Pass 9: both, /work/dev_s256.json, edit E2 alpha=1 (q_proj 5,11,17, all heads, remove u=norm(gq*gk*K[:,163])). Hypothesis: more reading -> better binding -> accuracy up.

Pass 9 result: E2 alpha=1: acc 11.7->2.7, diff -9.0 SE2.1, w2r 4, r2w 27, cut 12%->32%. Refuted: full sink removal. Copy in /work/e2_a1. Edits dropped.

E2 damage is immediate (0 identical, median divergence <40 chars), general comprehension. Style mix little changed.
Pass 10: both, /work/dev_s256.json, edit E3: same as E2 with alpha=-0.5 (amplify salience direction, stronger sink), q_proj 5,11,17 all heads.

Pass 10 result: E3 alpha=-0.5: acc 11.7->2.0, diff -9.8 SE2.1, w2r 3, r2w 28, cut 55%. Both directions hurt: original near a peak on salience axis. BOS sink line CLOSED. Edits dropped.

Free: last-mile line closed (36 gold-in-text not in problem, mostly extra final step/wrong last op, no repeated mechanism). Position-selective neuron search: no exclusive LP neuron in L16-17; L17 n395 is a line-start detector (unsafe at +8), best units reach half-level on ~4% of tokens.
Pass 11: both, /work/dev_s256.json, edit E4: down_proj L17 col 589 += dy/a_lp (a_lp=.331), dy min-norm writing +5 logits on Let(6481)-Here(8291) at last prompt token through post-ff gain and final gain. Purpose: causal test of opener style; flips analysed within paired. Limited power, side effect: line starts get up to +2.5.

Pass 11 result: E4 flipped 121/122 Here->Let. Overall -0.8 SE1.5, cut 12->23%. Flipped acc 9.9->8.3 (d -1.7 SE2.6), finished 10.7->10.5: opener association was selection. Opener style CLOSED causally. E4 dropped.
Set /work/dev_c256.json = DEV[512:768], disjoint from dev_a512 (selection data) and holdout.
Pass 12: both, /work/dev_c256.json, edit E5: zero down_proj columns of 34 wrong-associated units (|t|>=3.9 within style x length strata, LP or MA, sign(t)!=sign(mean)); units (layer,col) [(0, 179), (0, 733), (0, 1253), (1, 17), (1, 1147), (2, 57), (2, 115), (2, 697), (2, 1777), (2, 1802), (5, 1101), (5, 1501), (6, 744), (6, 1608), (7, 200), (7, 1971), (7, 1988), (8, 100), (8, 851), (8, 1451), (8, 1839), (9, 1631), (10, 788), (11, 1117), (11, 1659), (11, 1895), (12, 10), (12, 364), (14, 295), (14, 720), (14, 887), (15, 1573), (15, 1612), (17, 811)]
E5 size per layer: share of mean output .0001-.0126, share of matrix norm .022-.052. Verified against originals before the pass: only these columns differ, all zero, L17 c589 back to original.
Closed: opener (association and causal, E4), arithmetic, post-#### copy, grounding, reservoir c163 differential, gate-outlier switches, RMS dead coords, strong negative gains (E1), BOS sink both directions (E2/E3), last-mile, exclusive last-prompt neurons.
Plan: run pass 12. Clear gain -> split the 34 units and retest on fresh DEV. No gain -> close the correct-vs-wrong unit differential as causal, with the caveat that E5 is small. Original stays the fallback submission.

Pass 12 result: E5 acc 5.9->8.6, diff +2.7 SE1.7, w2r 13, r2w 6, cut 12.9->13.7%. Suggestive only (1.6 SE). Copy in /work/best_1. Next: read the changes, then replicate on untouched DEV.

Free: E5 on c256 changed 204/256 texts (median divergence 24% into text), 10 openers changed, no shared step in gains. Caution: c256 baseline 0.059 vs a512 0.125, z 3.2, similar reference length: the +2.7 may be regression to the mean, since any path reshuffle could recover accuracy on an unlucky slice.
Set /work/dev_d256.json = DEV[768:1024], no generation yet, disjoint from a512, c256, holdout.
Pass 13: both, /work/dev_d256.json, edit E5 unchanged (same 34 down_proj columns zeroed). Purpose: replicate on a fresh slice.

Pass 13 result: E5 on d256 9.8->8.6, diff -1.2 SE1.6, w2r 7, r2w 10, cut 14.1->10.9%. Pooled c256+d256 diff +0.8 SE1.2 (w2r 20, r2w 16). Not replicated; pass 12 gain fits regression to the mean on a low-baseline slice. E5 CLOSED, correct-vs-wrong unit line closed (caveat: 34 units only). Edits dropped. /work/best_1 holds E5 but is NOT a candidate.
Plan: audit which tensor families got analysis in my blocks, then test untested ones.

Free: dev_b256 == dev_c256 (same ids). d256 clean. Four 256 slices original 30/34/15/25 correct, chi2 p .034, pooled 10.2%. Untested causally: sliding-window heads, near-constant units, low-rank directions shared across layers. Running weight scan.

Free: reservoir audit. Residual energy dominated by c400 (early), c163 and c87 (from idx 6); c163 reaches ~23800 at idx 18. Top write directions of o_proj and down_proj are these coordinates, so the L1 o_proj outlier is reservoir, natural. E1 was confounded: L3c97, L6c129, L7c400 are reservoir coords (rank 1-3). 'Sign-flip gains load-bearing' applied only to the whole set.
Pass 14: both, /work/dev_s256.json (same problems as E1), edit E6: restore 7 non-reservoir gains to neighbour-layer mean weight, strength 1: [('post_attention_layernorm', 4, 118), ('post_attention_layernorm', 6, 365), ('post_attention_layernorm', 10, 419), ('post_attention_layernorm', 15, 122), ('post_attention_layernorm', 15, 605), ('post_feedforward_layernorm', 15, 43), ('post_feedforward_layernorm', 16, 353)].

Pass 14 result: E6 acc 11.7->3.9, diff -7.8 SE2.2, w2r 7, r2w 27, cut 12->58%. Confound idea refuted: non-reservoir sign-flip and muted gains are harmful to restore too. Gain line CLOSED. Edits dropped.

Free: E6 changed all texts at median char 5, finished-both acc 15.3->5.1, general damage. float16 overflow confirmed: non-finite values only at c163, 119 of 202 at BOS. Shared gate/up read direction L12-17: var about 1-2x average dir, not a top PC, word-vs-nonword axis; digit tokens OPEN gates, digit activity equals word. CLOSED as routing, no pass. No under-driven layer (active 78-86% from L2). Near-constant units 0-4 per layer. Collective constant MLP write 50-75% of non-reservoir energy from L6.
Plan: screen candidates with teacher-forced reference logprobs on /work/tf1_texts.json against pass 2, generate only for those that raise math logprob.

Note: token_logprobs[r,t] scores token t+1, so align with input_ids[:,1:].
Pass 15: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, compared with pass 2. Edit S1: zero down_proj columns of 210 gate-outlier units (gate row norm >= 3x layer median), list in /work/s1_units.json.

Pass 15 result: S1 refs_all +0.014 SE.004 (from newlines +.098), digit -.011, math -.023, punct -.074, others -.045; top-1 digit 83.96->82.54, down everywhere except space. No math-specific gain. Gate-outlier line CLOSED, no generation pass. Edits dropped. TF screen resolves about .005.

Free: head roles from pass 6 (2 texts only): most problem mass from answer positions L5h2 .27, L11h2 .33, L17h0 .31; least L11h0/h1 .02-.03, L17h1 .05.
Pass 16: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S2: o_proj columns of heads L5h2, L11h2, L17h0 multiplied by 1.5.

Pass 16 result: S2 (x1.5) refs_all -.062 SE.003, digit -.043, math -.253, others +.112; digit top-1 -.68 pts; 149/800 refs improved. Reading prediction refuted at this strength.
Pass 17: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S2b: same heads L5h2, L11h2, L17h0 o_proj columns x0.67.

Pass 17 result: S2b (x0.67) refs_all +.014 SE.003 (from math +.151), digit -.009, word -.008, others -.066; top-1 digit -.43, word -.99, math +.4. Digit top-1 falls both ways, peak near original; math symbols move monotonically, style. Head-balance line CLOSED. Edits dropped.
Pass 18: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 14,17,18, vs pass 2. Edit S5: down_proj row 163 x0.5 at L10, L13, L16 (reservoir level down). Metric: top-1, since final norm sharpens logits.
CORRECTION to Pass 18 edit: down_proj row 163 alone gives only about -15%, since c163 jumps at L5, L7, L10, L13 come mainly from attention. Final S5 = row 163 x0.5 in down_proj at L10, L13, L16 AND in o_proj at L10, L13. Predicted c163 at idx 18 about -29%. Inference to test: attention to BOS feeds the reservoir, so E2/E3 may be confounded with reservoir level.

Pass 18 result: S5 c163 at idx 18 25344->19968 (-21 pct, predicted -29), signal share .489->.578 (+18 pct, predicted +40; per-token signal already about half of rms). Top-1 all -.46, digit -.50, math +.54, space -1.13, punct -.95. refs_all -.241 but sharper logits confound logprob. Attenuation not the limit.
Pass 19: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 14,17,18, vs pass 2. Edit S5b: row 163 x1.5 in down_proj L10, L13, L16 and o_proj L10, L13.

Pass 19 result: S5b c163 +18 pct, signal share .489->.427. refs_all +.146 (softer logits), top-1 digit -.28, math -.52, space +1.5; battery digit -.88. Digit falls both ways, reservoir-level line CLOSED. Level alone moves top choices little, so E2/E3 damage likely came from moving attention (indirect). Edits dropped.
Pass 20: teacher_forced ORIGINAL, /work/tf20_texts.json, token_logprobs + top5. No edit. 250 refs (tf1 first 250) x 4 prompts: C0 eval, C1 two worked examples (tf1 idx [700, 701]), C2 no instruction, C3 careful-reading instruction. Purpose: does any state read and bind better?

Pass 20 result (original, 250 refs x 4 prompts): digit top-1 C0 83.93; change C1 examples -1.09 SE.31, C2 no instr -.08 SE.21, C3 careful -.10 SE.14. Every digit right about 6 pct in all. No better reading state reached, no justified steering target. Copied digits 81.5 harder than computed 84.5: binding is the bottleneck. Limited to 3 prompts on reference-style answers.

Free: math vs non-math at last prompt token: 500-1200 of 2048 units per layer |t|>6, confounded by prompt length (non-math questions short). Added same-problem rewrite control. Score = min(|t G1 vs G4|, paired |t G1 vs G2|), same sign, unit more active in G1 than both.
Pass 21: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S6: zero down_proj columns of 360 solve-math units (up to 20 per layer, score >= 6), list in /work/s6_units.json.

Pass 21 result: S6 (zero 360 solve-math units, about 10 pct of each matrix norm) refs_all -.004, digit +.002, punct -.187; top-1 digit -.48, math -.61, space -.99; battery digit -.8. No gain. Math-context units removal CLOSED. Solve-math state is redundant: 240-640 units per layer pass the double control.
Pass 22: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S6b: same 360 down_proj columns x2.
Plan after this: back off E6 to strength .2, then small-alpha salience, then units less active in solve-math.


Pass 22 result: S6b (360 units x2) refs_all -.030 SE.002, digit -.051, math -.055, punct +.157; top-1 digit -.99, math 0, space +1.1, word -.68; battery digit 77.2 (S6 77.0). Digit falls both ways (x0: -.48, x2: -.99), peak at original. Math-context units line CLOSED both directions. Edits dropped.

Free: q/k gains = 1+w; dead dims near w=-1, k outliers up to 30x; salience energy concentrated in top gain dims at L2, L5, L11. Pass 6 (2 texts only): reading heads diffuse, effective non-BOS positions L17h0 41, L11h2 25, L5h2 10; h1 heads 3-9. 
REFUSED, no pass used: S7 (q_norm gain x1.5 at L5, L11, L17 plus salience hold). save_edit: only rescales the norm's whole scale. Whole-norm rescale is out of scope; NOT attempting near-whole subsets of dims to get under the check. Whole-layer attention temperature remains UNTESTED. Mention in final report.

Free: key dim 203 is top k gain in 4 layers (p .0005 under shuffle) but test was post hoc, null too generous; same-index K row alignment weak and general (.037 vs .034, strongest dims 128, 0, 1), 203 unexceptional; gq*gk product 2-10 near layer median 2-7 (trade-off, near 0 at L16); reads reservoir coords. CLOSED, not a planted channel.
Free: keys 55-87 pct constant after gains (L17 .27); constant mostly non-reservoir (L5 .58 vs .00 from c163). Norm acts before gains, so raw share is the budget: L5 .41, L11 .34, L17 .38 from non-reservoir.
Pass 23: teacher_forced EDITED, /work/tf6_texts.json (2 texts), attn 5,11,17 + token_logprobs, vs pass 6. Manipulation check only. Edit K0 beta=1: k_proj L5,L11,L17, remove unit mean-key direction (non-reservoir columns, pass 7 sample of 40 texts, non-BOS) from all columns except 163. Same hypothesis as the REFUSED S7 (reads too diffuse), by a different and allowed edit; state this in final report. Expect: effective non-BOS positions down, BOS share about flat.

Pass 23 result: K0 beta=1 sharpened reads (effective positions L17h0 41->23, L11h2 25->11, L5h2 10->4.5) but BOS share jumped (L5 h1-3 ~.45->.96, L11h2 .24->.48). Manipulation failed on BOS condition, confounded with E3 sink. No full screen. Own-text logprob -.06->-.39, -.18->-1.02 (rough). Constant key carries a positive offset against BOS, so not simply wasted budget. Edits dropped.

Free: constant key outside span of mean queries: L5 .63, L11 .91, L17 .96 of its energy; boost if only that removed 1.16, 1.20, 1.25.
Pass 24: teacher_forced EDITED, /work/tf6_texts.json (2 texts), attn 5,11,17 + token_logprobs, vs pass 6. Manipulation check. Edit K1 beta=1: k_proj L5,L11,L17, remove unit direction n = part of mean key (non-reservoir columns) orthogonal to span of 4 gain-weighted mean queries, from all columns except 163. Same hypothesis as refused S7, different allowed edit. Pass condition: reads sharpen AND BOS share near pass 6. Big BOS jump closes the line.

Pass 24 result: K1 beta=1 manipulation PASSED: reads sharper (L17 h0 41->19, h2 40->10, h3 38->11, L11h2 25->12), BOS no jump, moderately lower at L17 (h2 .67->.38, h3 .56->.25, h0 .45->.30). Side effect: h1 pointer heads broader (L11 7->18, L17 9->20). Own-text logprob -.06->-.12, -.18->-.56 (rough). Query-aligned part of constant key balances BOS, confirmed.
Pass 25: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit K1 unchanged (k_proj L5,L11,L17, beta=1). Digit top-1 decides; split copied vs computed digits after. Rise -> generation on fresh DEV. Fall -> softening check, then close.

Pass 25 result: K1 beta=1 refs_all -.207 SE.007, digit -.247, math -.371, word -.229; top-1 all -3.43, digit -5.79 (83.96->78.17), math -5.12, punct -6.24, space +3.86; battery digit 77.8->70.75; 108/800 refs improved. Sharper reads hurt broadly, even isolated arithmetic. Edits dropped.

Copied vs computed K1: copied 82.76->75.91 (-6.85 SE.50), computed 84.59->79.37 (-5.22 SE.33). Copied hit harder: binding prediction REFUTED at beta=1.
Pass 26: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit K1 beta=-0.5 (ADD along n, softer reads), k_proj L5,L11,L17, columns except 163. Symmetric-peak prediction about -1.4 digit top-1; a rise means the peak is softer than original. Axis closes after this pass either way unless it rises.

Pass 26 result: K1 beta=-.5 refs_all -.065 SE.004, digit -.031; top-1 digit -1.25 (predicted -1.4), math -1.82, word -1.91; copied -2.0 SE.37, computed -.86 SE.22; battery digit 77.8->77.0. Curve through beta 1, 0, -.5 peaks at beta about -.02. Sharpness axis CLOSED both directions at L5, L11, L17 (edit route; whole-norm rescale refused and not used). Edits dropped.
Pattern: salience, head balance, reservoir, gains, math-context units, sharpness all peak at original, no axis shows gain in either direction.
Plan: per-layer margin ledger, then causal digit-write test on any layer with negative mean margin contribution; then units switched off during math.

Free: margin ledger rebuilt, sum check 1.0. idx14 (L13) retraction r -.63 but net HELPFUL (fail +.81, net flips +26 per 1000 digits). L13 retraction claim CLOSED, no pass. No layer net harmful (worst -4 to -8 per 1000: idx2,6,8,11). Digit decision at idx12,13,16 = L11,L12,L15 (+153,+23,+15 per 1000); earlier layers near 0. Failing digits are pushed wrong there, reads as weak decision.

Free: units off during math rare and scattered: 41 off vs 673 on at ratio .25 (11 vs 281 at .10), last prompt token only. Switched-off gate line CLOSED, no pass.
Pass 27: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S8: zero down_proj columns of 133 math-only units (score>=6, active in math >= layer median, both controls quiet), list /work/s8_units.json; overlap with S6 27. Direct test for a math-gated suppressor. Rise = suppressor; fall = capability; flat = redundant.

CORRECTION: '41 off vs 673 on' above was NOT a fair comparison: the on count used the lower control only (quiet in at least one control). Fair mirror counts, both controls required: {'off25': 41, 'on25': 19, 'off10': 11, 'on10': 4}. S8 is 133 units (both controls quiet, cutoff .5), as logged; the 673 figure and the 16:1 ratio I reported in my explanation were wrong.

CORRECTION 2: fair counts are 41 off vs 19 on (.25), 11 vs 4 (.10), a mild 2:1 lean toward OFF, not 16:1 toward on. My claim that this runs opposite to a planted gate was wrong and the closure 'Switched-off gate line CLOSED' is WITHDRAWN. Line REOPENED: counts small and scattered, neither show nor exclude a gate. Un-silencing test for the 41 units queued after S8. S8 unchanged: 133 units, loose cutoff .5, pass 27 still planned as logged.

Pass 27 result: S8 (133 math-only units zeroed) refs_all -.002 SE.002, digit -.009, math -.059; top-1 digit -.34, math -.61; copied -.12 SE.20, computed -.46 SE.15; battery digit +.16. Not suppressors, mildly useful and redundant. Math-only unit line CLOSED. Edits dropped.

Free: 41 off units quiet at last prompt token only; over answer tokens math/rewrite activity ~.6-1.85 in most layers (exceptions L12 .53 n1, L14 .04 n1, L15 .33 n4). Missing write <=4.5 pct of layer mean. Constant write wrong edit. Off-unit line CLOSED on this basis; prompt-boundary silencing untested, 6 units unexplored.

Free: digit evidence route split (pass 3, own greedy answers, 3878 digit positions): evidence ~0 through L10, then L11 +3.34 (attn 3.03), L12 +4.16 (attn 2.94, mlp 1.22), L13 -.44 (attn -1.14 against chosen digit at 84 pct positions, mlp +.70), L15 +2.27 (attn 1.45). Attention carries 7.2 of 10.9. Coordinate spread wide (ipr 117-200), not a planted coordinate channel. mlp_act_l = model.layers[l]; L maps hidden_l to hidden_l+1.
Unscreened still: sliding-window heads (attention analysis so far only L5,L11,L17), steering edits (constant vector into residual, none tried), v_proj.
Pass 28: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 13,14,18, vs pass 2. Edit D1: o_proj L13 only, O -= beta*GU S^-1 (GU)^T O, beta=1, GU=post-attn gain * U, U = orthonormal 9-dim which-digit subspace (gn*(E_digit-mean E_digit), reservoir coords 163,400,87 zeroed). Prediction from ledger: digit top-1 falls up to 2.6. Rise = retraction harmful, first positive.

Pass 28 result: D1 (L13 o_proj digit-subspace write removed) refs_all -.014, digit -.041 SE.003; top-1 all -.08, digit -.21 (83.96->83.75), math -.18; battery digit unchanged 77.8. Ledger predicted up to -2.6: association overstated more than 10x. Retraction slightly useful, not a brake. L13 retraction CLOSED causally. Evidence scale barely moves top choice (fails not near ties). Edits dropped.
Free: o_proj head norms: no simply muted head. L11 h2,h3 gain-weighted 5000 vs h0,h1 1000, spread 5.9x vs <=2.5x other layers; digit-write energy 25x h2,h3 over h0,h1, raw col energy 3x.

D1 split: copied -.13 SE.23, computed -.25 SE.12. Free: write-hole scan on o_proj heads vs 40 control sets of 10 frequent tokens: L11 h0,h1 weak for ALL classes and digit-enriched (+2.8,+8.2 SD); pointer-head disconnection REFUTED. Digit depleted L4-9 (-2 to -4 SD), enriched L10-13,15,17, matches ledger.
CORRECTION to 'Pattern ... all peak at original': for salience (E2 alpha 1, E3 alpha -.5) and gains (E1,E6 strength 1) that claim rests only on large strengths. Back-off to small strengths was planned after pass 22 and never run. Peak location NOT established on those two axes. Queue: E2 small alpha, E3 small alpha, E6 strength ~.25.

Free: read-side (q,k,v,gate,up) and down_proj digit hole scan: smooth, read enriched L3-9, write enriched L11-17, min -5.1 at L2 attention with mixed neighbours. Digit hole line CLOSED both sides.
Pass 29: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit E2s alpha=.25: q_proj L5,L11,L17 all heads, Q_0 + .25*(Q_e2a1 - Q_0), equals Q_h -= .25 u u^T Q_h. Back-off owed on salience axis; naive quadratic through generation points peaks near alpha .26, +1.3, not trusted (floor).

Pass 29 result: E2s alpha .25 refs_all -.036, digit -.037; top-1 digit -.35, math +.5, space +2.68, word -.51; battery digit 77.8->76.68; copied +.10 SE.25, computed -.59 SE.18. No rise, positive side falls gently. Edits dropped.
Pass 30: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit E3s alpha=-.15: q_proj L5,L11,L17 all heads, Q_0 -.15*(Q_e2a1 - Q_0), equals Q_h += .15 u u^T Q_h. Closes salience back-off.

Pass 30 result: E3s alpha -.15 refs_all -.014, digit -.010; top-1 digit -1.00, math -.74, space -1.06; battery digit 77.8->78.45; copied -1.60 SE.27, computed -.68 SE.16. Quadratic through -.15,0,.25 peaks near alpha .09, about +.16 digit, within noise. Salience axis CLOSED on close-range evidence.
Pass 31: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit E6s strength .25: same 7 gains as E6 (post_attn L4c118,L6c365,L10c419,L15c122,L15c605; post_ff L15c43,L16c353), w -> w + .25*(neighbour mean - w). Closes gains back-off.

Pass 31 result: E6s strength .25 refs_all +.019, digit logprob +.048, punct +.225; top-1 all -.28, digit -.38, math -1.14, punct +1.56; copied -.05 SE.17, computed -.55 SE.11; battery digit 77.8->77.56; 506/800 refs improved. Softer logits, not better decisions (same signature as S5b). Seven-gain axis CLOSED at close range.
Free: muted coordinates (|w|<.1 |neighbour mean|, |nb|>=30) cluster at L10: post_attn 13, post_ff 9, others <=5. Checking threshold artifact.

Free: L10 muted cluster mostly threshold artifact: scale-free post_attn L10 13->2. post_ff 9 at L10 are coords switching on at L11 (L9 ~12, L10 ~0, L11 ~270, L12 ~440), 5 with gain ~0; L2,L4 scale-free counts are w~0 i.e. gain ~1, not muting. CLOSED, no pass. Note high-gain coordinate group from L11 includes 149,518 (ledger top coords).
Plan: steering, none tried yet. Math-state reversal first.

Free: math-solve state at last prompt token exists: cos(G1-G2,G1-G4) .4-.6 from idx 6, |d|/within-SD 1.3-1.9 through idx 10, share of state norm .2-.4, write share rho .26-.37 at idx 8-12.
Pass 32: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 11, vs pass 2. Edit T1 (steering): down_proj L10 column 1015 += beta*(t/g)*(rms/a_j), t=d=mean(G1)-.5(mean(G2)+mean(G4)) at idx 11 last prompt token, beta=-1 (writes -d, moves math state to control midpoint); reservoir and low-gain coords excluded. Rise = math state carries handicap; fall = math state is capability.

Pass 32 result: T1 beta=-1 manipulation exact (LP shift 619 vs |d| 620, cos .999, projection .997). refs_all -.115, digit -.083; top-1 digit -1.05, math -.52, space +6.47, punct +2.62, word -1.95; copied -.44 SE.28, computed -1.37 SE.21; battery digit 77.8->77.24, battery all 78.8->83.2. Math state not a brake, mildly useful, mostly format. Edits dropped.
Pass 33: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit T1b: same as T1 with beta=+1 (writes +d, amplifies math state), down_proj L10 col 1015.

Pass 33 result: T1b beta=+1 refs_all -.003, digit -.016; top-1 all -1.61, digit -1.81, math -1.31, space -2.14, word -1.26; copied -1.24 SE.28, computed -2.11 SE.24; battery digit 76.92. Parabola through beta -1,0,+1 peaks near -.13, about +.03. Math-state steering axis CLOSED at L10, exact manipulation. Edits dropped.

Free: copy matrix (v_proj -> head o_proj -> digit logit, measured mean hidden per digit, pass 3, double-centred, diag mean / offdiag sd): copiers L11h2 +8.56, L11h0 +4.40, L12h1 +9.62, L12h2 +4.83, L12h3 +3.60, L12h0 +3.16, L13h2 +3.86, L14h2 +3.81, L17h3 +4.48; ANTI-copiers L11h3 -8.67, L13h3 -9.29, L17h2 -5.27, L13h0 -3.44. Shift bands all |z|<1: no scramble. L6-9, L15, L16 quiet.
Free: pass 6 (2 texts, 80 digit positions) L11: h2 correct-digit mass .376 other .153 selectivity .71 vs chance .27; h3 .145/.183 selectivity .44; h2,h3 attention pattern corr .70, correct mass corr .76: same sources, opposite write. Rough margin h2 alone 1.91, h2+h3 2.24: contrast enhancement, not suppression, prediction only.
Pass 34: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit D2: o_proj L11, columns of head 3 only, remove write into U9 (beta=1). Fall = contrast enhancement; rise, esp. copied = brake.

Pass 34 result: D2 (L11h3 digit write removed) refs_all -.016, digit -.038; top-1 all -.31, digit -.73, math -.16; copied -.55 SE.27, computed -.81 SE.17; battery unchanged. Anti-copy head helps: contrast enhancement supported, not a brake. Edit dropped.
Pass 35: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit D2b: o_proj L11, head 3 columns, beta=-.5 (digit write x1.5). Rise = contrast under-strength; fall = peak at original.


Pass 35 result: D2b beta -.5 (L11h3 digit write x1.5) refs_all -.001, digit -.013, math +.019; top-1 all +.01, digit -.24, math +.19, space +.5; copied -.27 SE.16, computed -.22 SE.12; battery digit 77.8->77.4. Both directions lose (D2 -.73, D2b -.24): peak near original. L11h3 contrast axis CLOSED, not a brake. Edit dropped.

Free: provenance scan: all 236 tensors BF16, all round-trip exactly, no tensor stands apart. Route closed, does not show weights untouched.
Pass 36: teacher_forced ORIGINAL, /work/tf36_texts.json, token_logprobs + top5. No edit. First 250 refs x 4 wrappers: C0 eval, W1 raw Question/Answer, W2 raw problem + Solution:, W3 chat template + expert accuracy preface. W1,W2 no turn tokens, trailing end_of_turn stripped. Purpose: context-keyed limit? Rule fixed before data: better state only if digit top-1 rises >=1 point with |z|>3 vs C0.

Pass 36 result (original, 250 refs x 4 wrappers, 6821 digits): digit top-1 C0 83.93; W1 raw Q/A -.16 SE.28, W2 raw solution -.05 SE.24, W3 expert preface -.05 SE.19; copied, computed, every-digit flat; dlp W1 +.04, W2 +.03, no top-1 change. Rule not met: no better state, no context-keyed brake at digit level. Wrapper-context line CLOSED. Caution: flatness suggests digit top-1 on refs is dominated by local text, so it may be blind to plan-level effects. Earlier TF closures rest on it. Rescoring saved TF passes at plan-sensitive positions, free.

Free: rescored 18 TF edit screens at plan-sensitive positions vs pass 2: base num1 66.38, num2 74.88, num3-4 77.12, later 78.09, cont 91.55, op12 46.39. No edit rises >=1 with z>=3 at early positions. Power limited: num1,num2 cells n=800 SE ~.7. Earlier closures stand against large plan gains, not small ones. Number choice and operator, not continuation digits, are the weak decisions (style caveat).
Free: correct-minus-wrong split-half (pass 1, strata reference length x opener, cut failures dropped): LP cos .38-.46 mid layers, MA .31-.37, best LP idx10 z4.3, MA idx11 z4.0; idx1 LP cos .695 shows problem-content confound, so reliability is not causality.
Pass 37: both, /work/dev_e256.json (DEV[1024:1280], fresh), activations layers [11], mlp false. Edit C1 beta=+2 (steering): down_proj L10 column 1015 += beta*(t/g)/(a/rms), t = stratified mean correct - mean wrong of hidden_mean_answer idx 11 (dev_a512, /work/c1_target_idx11.npy), mean a/rms 31.122 over finite pass 3 answer positions, reservoir 163,400,87 and low-gain coords excluded. Rise >2 SE = replicate on DEV[1280:1500] plus dose check. Flat or fall = direction labels easy problems, not capability.

CORRECTION: all-NaN rows in pass 3 activations are padding (pad id -1, not 0), not float16 overflow: 2951 of 2951 beyond recorded length, none inside real text. C1 calibration a/rms 31.122 rests on all 5058 sampled real answer tokens, no selection bias.
Pass 37 result: C1 beta=+2 on dev_e256: acc 14.06->12.11, diff -1.95 SE2.03, w2r 11, r2w 16, cut 12.5->14.1. Edit dropped.
Pass 37 manipulation exact: LP projection .929 (expected .945) cos .996, answer mean 1.055 cos .988, 1.45-1.56 SD along t; 242/256 texts changed, median divergence .215. Strength adequate. Correct-minus-wrong at idx 11 is a correlate, not a lever. CLOSED for this direction, layer, strength; not all steering excluded.
Gap found: E2/E3, K1, S2 were joint edits on L5, L11, L17, so per-layer effects may have cancelled.
Pass 38: both, /work/dev_e256.json, activations none. Edit E2-L17 alpha=1: q_proj L17 only, all 4 heads, values from /work/e2_a1 (Q_h -= u u^T Q_h, u=norm(gq*gk*K[:,163])). Joint E2 was -9.0 SE2.1. Much better than joint = damage from earlier layers; clear rise = candidate, replicate on DEV[1280:1500]; large fall = sink needed at L17 too.
Pass 38 result: E2-L17 alpha=1 on dev_e256: acc 14.06->10.55, diff -3.52 SE2.02, w2r 9, r2w 18, cut 12.5->11.7. No gain. Milder than joint -9.0 (different slice), cut-offs normal vs joint 32 pct, so runaway answers came from L5/L11. L17 sink not hiding a gain. Edit dropped.
Free: L10 post_ff near-zero gains: exactly 149,374,403,518,546, all in high-gain L11 group (113 coords); same five L10 down_proj rows norm 2.06-2.18 vs median .58. Looked planted, is NOT: scan shows large rows with gain ~0 at nearly all layers, down_proj and o_proj (o_proj L8 66 rows). Five rows share no direction (cos .008, participation 456 units), row norm vs gain is a continuum (575, 412, 618, 398). Mechanism: sink rows, RMSNorm forces unit rms so layer writes less by putting energy in zero-gain rows. L10 five take 51 pct of output energy at answer tokens, boost 1.45 if removed. Share LOWER in math: LP G1 .258, G2 .300, G3 .407, G4 .363; answer .542 vs .586-.593. Not math-gated at L10.
Free: sink rows all layers down_proj (pass 4): L0,L1,L8 flagged math above both controls; within token class excess gone (L1 word G1 .562 vs G3,G4 .585-.589; digit +.03; L0, L8 about .01; L8 answer lower in math). Math-gated attenuator in down_proj CLOSED, no pass. Late layers more open in math. Attention sink rows unmeasured, head outputs not returned. Explains E1,E6 damage (mechanistic reading): restoring gains on sink-like large rows made them loud writers.
Gap: S2 scaled head columns, which under RMSNorm is head balance, not volume. Layer write volume untested.
Pass 39: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V1: L10 down_proj rows 149,374,403,518,546 zeroed (sink rows, post_ff gain ~0), L10 MLP write volume up ~1.45 at answer tokens. Rise esp at number starts = under-volume, go to generation. Fall = peak at original, closes.
Pass 39 result: V1 (L10 sink rows zeroed) refs_all -.106, digit -.127, math +.121; top-1 all -1.63, digit -3.28, math +.48, word -.70; copied -2.52 SE.39, computed -3.67 SE.25; battery digit 77.8->75.3; refs improved 117/800; number starts num1 -.75 z-1.0, num2 -1.0 z-1.0, num3-4 -3.44 z-5.1, later -2.57 z-8.7, cont -4.16 z-16, op12 +.74 z1.2. Sink rows load-bearing, causal. Edit dropped next.
Pass 40: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V1b: L10 down_proj rows 149,374,403,518,546 x1.3, L10 MLP write volume about x.86. Symmetric peak predicts digit top-1 about -.6. Clear rise = over-volume, first positive. Fall near prediction = optimum, MLP volume axis closes at L10.
Pass 40 result: V1b (L10 sink rows x1.3, volume about x.86) refs_all +.0026 SE.0011, digit -.001, math -.071, others +.060; top-1 all -.15, digit +.01, math -.41, punct +.5, word -.34; copied +.01 SE.18, computed +.01 SE.13; battery digit 77.8->78.04; 424/800 improved; num1 +.5 z1.2, num2 -.5, num3-4 -.38, cont +.06, op12 -.29. Symmetric prediction -.6 refuted: asymmetric, steep up (-3.28), flat down. Three-point parabola peak ~.93 volume, ~+.1, within noise. Plateau = redundancy, no measured gain, not a candidate on this evidence. Edit dropped.
Pass 41: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V2: o_proj sink rows zeroed, L11 [61, 319], L12 [24, 629] (norm >2x median, |gain| < max(.25,.05 median)), attention write volume up at L11, L12. Rise esp copied and number starts = under-volume, generation. Fall = tuned.
Pass 41 result: V2 (o_proj sink rows zeroed L11 61,319; L12 24,629) refs_all -.090, digit -.158; top-1 all -1.13, digit -3.17, math -1.7, space +3.24, punct +1.44, word -1.54; copied -.75 SE.35, computed -4.45 SE.27; battery digit 77.8->70.03; 179/800 improved; num1 -.38, num2 -2.25 z-2.3, num3-4 -6.75 z-8.5, later -4.06, cont -2.23, op12 -1.11. Rise prediction refuted. Attention volume up hurts arithmetic far more than copying: sink rows regulate computation at L11,L12. Edits dropped next.
Pass 42: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V2b: o_proj rows L11 [61, 319], L12 [24, 629] x1.3, attention volume down (size unmeasured). Rise esp computed and battery = first positive, generation. Flat or fall closes volume family.
Pass 42 result: V2b (o_proj sink rows L11,L12 x1.3) refs_all -.022, digit -.017, math -.024; top-1 all -.14, digit -.07, math +.37, space -.84; copied -.18 SE.18, computed -.01 SE.12; battery digit 77.8->78.29, all 79.08; 179/800 improved; num1 +.38, num2 +.12, num3-4 -.12, later +.18 z1.1, cont -.31 z-3.0, op12 +.52 z1.6. Flat, same asymmetry as L10. Rule not met. Battery direction consistent across V2,V2b (70.03, 77.8, 78.29), not significant: dose check.
Pass 43: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V2c: o_proj rows L11 [61, 319], L12 [24, 629] x2. Real trend = battery well above 78.29 and digit top-1 clearly positive. Flat or fall closes volume family, then final phase. Keep at least 3 passes for HOLDOUT.

Pass 43 result: V2c (o_proj sink rows L11 61,319; L12 24,629 x2) refs_all -.116, digit -.099; top-1 all -.91, digit -1.19, math +.44, space -2.14; copied -.84 SE.31, computed -1.37 SE.21; battery digit 77.8->76.36, all 78.05; 57/800 improved; num1 0, num2 0, num3-4 -.88, later -1.39 z-5.6, cont -1.26 z-7.4, op12 +1.18 z2.4. Trend rule refuted: x1.3 was flat top or noise. Volume family CLOSED at L10 MLP and L11,L12 attention. op12 rise is one cell of many, with broad losses, not a lead by itself. Edits dropped. 7 passes left, final phase next.

Final-phase review: read whole notebook. No edit has a replicated generation gain, HOLDOUT unused, no candidate. TF digit screens may be blind to plan-level effects (pass 36 caution). Unexamined: RoPE band structure of q/k, v_proj, sliding-window heads. Free next: weight-only RoPE band audit by head and layer. Anomaly = causal screen; smooth = close, then generation checks of least damaging edits. Refused S7 stays out of bounds, band audit is not a route to it.

Free: RoPE band audit (gq,gk weighted q-k coupling, reservoir excluded): reading heads 94-97 pct in period >512 vs 68 pct of dims, pointer heads 77-79; local mean .706 sd .103, no head beyond |z| 2.6 among 60. Smooth, no missing content read. RoPE band line CLOSED, no pass.

Free: plan index (num1+num2+op12, 2956 positions, base 59.51) over 23 TF screens, sorted: V2c +.54 z1.5 digit -1.19; S5 +.51 z1.8 digit -.50; V2b +.37 z1.5 digit -.07; D2b +.14; rest lower; V1b -.14. Post hoc, none z>=2, sets test order only. V2b best with flat digits. Joint V1b+V2b not supported.
Volume family stays closed on TF digit evidence. V2b reopened for GENERATION only, because TF digit metric is blind to plan effects.
Pass 44: both, /work/dev_f256.json (DEV[1280:1500] fresh + DEV[1244:1280]), no activations. Edit V2b: o_proj L11 rows 61,319 and L12 rows 24,629 x1.3, copy /work/v2b. Rule fixed before data: diff >= +1 SE and w2r > r2w = replicate on another DEV slice; else drop and test S5 in generation. Original stays fallback. HOLDOUT unused, keep >=3 passes.

Pass 44 result: V2b generation on dev_f256: acc 14.06->11.33, diff -2.73 SE1.95, w2r 9, r2w 16, cut 12.1->17.2, mean tokens 282->309. Rule not met, V2b dropped. Flat TF digits and plan index +.37 did not carry to generation. Volume family CLOSED in generation too for this edit.
Pass 45: both, /work/dev_f256.json, no activations. Edit S5: row 163 x0.5 in down_proj L10,L13,L16 and o_proj L10,L13. Rule fixed before data: diff >= +1 SE and w2r > r2w = replicate on another DEV slice; else drop. Original stays fallback. HOLDOUT unused, keep >=3 passes.

Pass 45 result: S5 generation on dev_f256: acc 14.06->10.16, diff -3.91 SE1.82, w2r 6, r2w 16, cut 12.1->11.7. Rule not met, S5 dropped, digit-for-plan trade idea closed.
CAUTION on slices: pooled original over all 1500 DEV = 11.30 pct; e256 and f256 both 14.06. Mild edits there scored 12.11, 10.55, 11.33, 10.16 (mean 11.04), near pooled baseline: their losses are largely regression to the mean on high-baseline slices sharing one original sample. Report as no gain, not as proven harm. Mirror of E5 +2.7 on low-baseline c256. Passes 44 and 45 share one original sample, not independent.
Free: within-problem discordant pairs (151 pairs, 125 problems, mild passes 11,12,13,37,38,44,45): right paths shorter 235 vs 307 tok, fewer lines, cut 7 vs 19 pct, |t| 2-2.7 before correction; r2w 90 vs w2r 61 confounds with edit direction, loops inflate length.

Free: balanced uncut pairs (114; w2r 42, r2w 72): n_tokens -12 z-1.74, lines z-1.9, numbers z-1.74, latex, bold, step, let |z|<.6. Earlier contrast was mainly loops plus edit direction. Style features line CLOSED. Better design: stratify DEV test sets by known original outcome to match 11.3 pct, removing regression to mean.

Free: first divergence of 151 discordant pairs: word-word 100, digit-digit 10, symbol-symbol 9, others 32; median token 31. Outcome-changing paths mostly split at words, base rate caveat. Supports caution that digit screens miss plan effects.
Pass 46: teacher_forced ORIGINAL, no edit, /work/tf46_texts.json (meta tf46_meta.json), token_logprobs + hidden 6,11,14,17. Within-problem state contrast. Window: from first divergence to shorter path end. Balanced w2r and r2w, reservoir coords 163,400,87 excluded. Rules fixed before data: reliability needs mean split-half cos by problem >=.15 and above 95th pct of sign-flip null, else CLOSE; |cos| with C1 target at idx 11 >=.7 = covered by C1, no generation; both pass = one steering generation on DEV set stratified to 11.3 pct baseline. Remaining plan: 5 passes; if no candidate, HOLDOUT only for honest reporting; original stays fallback.

Pass 46 result: within-problem contrast. Raw reliability idx6 .275 (null p95 .268), idx11 .494 (.188), idx14 .344 (.159), idx17 .319 (.17); cos with C1 .179; divergence |gap| median .38 nats, near ties. Artifact checks at idx 11: composition fixed .462; drop last 25 tokens .204 (null .18); drop 25 + uncut pairs -.026 (null .214). Direction was endings and loops. No reliable within-problem state during the working. Line CLOSED, no generation pass, by rule stated before these checks.

Free: post-stratified to pi=.113 (removes slice baseline effect): C1 -.59 SE1.60, E4 -.66 SE1.32, E5 -.76 SE1.20, V2b -1.39 SE1.51, E2L17 -2.02 SE1.51, S5 -2.60 SE1.35; E1,E2,E3,E6 -7.4 to -9.4. All negative. become_right below neutral in all six (E5 .042 vs .051): perturbation costs slightly more than it gains, fits local optimum, each within noise of zero alone. E5 raw +.8 was baseline-flattered.
Decision rule fixed before pass 47: C1 stays frozen for HOLDOUT if pooled post-stratified estimate (passes 37+47) >= E5 -.76; else E5 frozen. HOLDOUT max 3 passes (750 ids). Submit an edit only if pooled HOLDOUT gain >= +2 SE; else original.

Pass 47: both, /work/dev_strat256.json (29 orig-correct + 227 orig-wrong, seed 20261104, pool DEV[512:1024]+DEV[1280:1500], excludes a512 and e256), no activations. Edit C1 beta=+2 rebuilt from block 178 recipe, copy /work/c1_edit: down_proj L10 col 1015 += beta*(t/g)/a_rms. Rule logged above: C1 stays frozen if pooled post-stratified (37+47) >= E5 -.76, else E5 frozen. Then up to 3 HOLDOUT passes.

Pass 47 result: C1 beta=+2 on dev_strat256 (baseline 11.33): acc 11.33->7.81, diff -3.52 SE1.78, w2r 6, r2w 15, cut 10.9->12.9. Pooled post-stratified 37+47 about -2; below E5 -.76, so by logged rule C1 out, E5 FROZEN for HOLDOUT. E4 -.66 SE1.32 was not in the rule and its flip analysis already showed no causal gain.
FROZEN: E5, 34 down_proj columns zeroed, /work/best_1. No further changes. HOLDOUT plan, passes 48-50: both, holdout_1.json (256), holdout_2.json (256), holdout_3.json (238), same edit, no activations. Primary: pooled paired diff over all 750. Submit E5 only if pooled gain >= +2 SE; else drop all edits and submit original. HOLDOUT never used before this.

Pass 48 result (HOLDOUT part 1, E5 frozen): acc 8.98->9.77, diff +.78 SE1.46, w2r 8, r2w 6, cut 13.7->12.1. Single part, not read alone. Pass 49: both, /work/holdout_2.json, same frozen E5, no activations.

Pass 49 result (HOLDOUT part 2, E5 frozen): acc 9.77->11.33, diff +1.56 SE1.46, w2r 9, r2w 5, cut 12.9->15.2. Interim parts 1-2: w2r 17, r2w 11, about +1.17 SE1.03, not a decision point. Pass 50: both, /work/holdout_3.json (238), same frozen E5, no activations. Rule unchanged: primary is raw pooled paired diff over all 750; submit E5 only if >= +2 SE; else drop all edits, submit original. Post-stratified figure reported beside it, secondary.

Pass 50 result (HOLDOUT part 3, E5 frozen): acc 11.76->9.24, diff -2.52 SE1.78, w2r 6, r2w 12, cut 13.4->8.8.
FINAL HOLDOUT pooled over 750: acc 10.13->10.13, diff 0.00 SE 0.90, w2r 23, r2w 23. Rule was submit E5 only at >= +2 SE. Not met. ALL EDITS DROPPED, /work/edited empty, ORIGINAL SUBMITTED. All 50 passes used.
