# Notes
## Sets
DEV=/work/dev_ids.json (1226), HOLDOUT=/work/holdout_ids.json (1024; hold_0..3.json, 256 each). Seeded shuffle 20261104, fixed before any generation. HOLDOUT never used for choosing/tuning.
## Pass log
pass1: original, generate, dev_512a.json, activations all layers + mlp. No edit exists.
pass2: TF original, no edit. texts=own 512 answers (gen_1) + 512 stripped refs (same problems). token_logprobs, topk 10, no hidden.
Findings so far: no weight anomaly by norms/spectra. Extraction not the cause. arithmetic ~5%/eq clean, multi-op worse; accuracy halves per step (~60%/step reliability). loops median period 48, gold in pre-loop only 10%.
pass3: TF original, attn_layers [5,11,17,23] on 2 texts (solved answer + failing 'fish' answer gsm8k_train_114). No edit.
Retrieval heads found (pass3, n=23 digit events): L17h0 top-attn matches next digit .83 vs chance .35, mass .23; L11h1 .65/.22.
pass4: TF on tf2_texts. EDIT: L17 o_proj head0 columns (0:256) *0 (ablation). Prediction: copy digits fall most.
pass4 result: L17h0 ablation: copy first -4.2pt top1, copy later -7.7, other later -5.0, result ~-0.5 => causal + specific number-copy head.
pass5: TF on tf2_texts. EDIT: L17 o_proj head0 columns *2 (boost). Prediction if capacity-limited: copy digits rise.
pass5 result: L17h0 x2: copy +0.3pt only, result -0.7, own solved retention 0.990 => head necessary but NOT limiting; dropped. No edits active.
Screen metric: copy/other FIRST digit on refs = 'which quantity next' decision.
pass6: TF on tf2_texts. EDIT: L11 o_proj head1 *2 AND L5 o_proj head2 *2 (candidate relation-reading heads). Prediction if limiting: copy/other first rise.
pass6 result: L11h1+L5h2 x2: prompt dlp +.030 but answer dlp -.021, copy/other first flat => reader heads NOT limiting. dropped. Head gain line closed.
pass7: TF original, no edit. texts=89 loop answers truncated to 4 cycles; hidden layers 4..26 sparse; contrast cycle1 vs later cycles at same token.

x1038 (huge near-const residual coord, 80-98% of energy L8+; preff weight -1): right/wrong/cut within ~5% at same layer, answer-mean CV 2-5% => NOT a failure mechanism, usable as near-constant source only.
Baseline gen_1: acc .193, cut 28%, wrong finished 269/512. Modes: heres n=230 acc .170 cut .226; let n=277 acc .217 cut .332; finished acc heres .213 let .324.

Modes controlled: stratified by ref steps let-heres +7.9+-3.4pt (observational). First-token gap Let-Here median .25, IQR -1..1.6; topk[i-1] predicts token[i]. Modes separate d=1.5 L5, 2.0 L12, 2.6 by L17 then flat. Selective last-prompt MLP units e.g. L17 n4587, L18 n2543.
pass8: TF original, no edit. texts=/work/tf8_texts.json (48 own answers, mode x cut, idx in tf8_idx.json); mlp_act [5,11,17,23,24], hidden [6,12,18,24,25]. Purpose: token-by-token selectivity of last-prompt units.

Calib checks: hidden l = after model.layers.(l-1) (cos .956); post-FF scale=(1+w) confirmed (proj .945). pass8 result: L23 n1446 a=2.57 rms elsewhere .07, single leak at blank line offset -29 (30%); L11 n4788 a=-1.42 rms elsewhere .08. L17 n4587 rejected (5.06 elsewhere).
pass9: BOTH on dev_c2 (256). EDIT=cand_let1: model.layers.23.mlp.down_proj column 1446 += alpha*d24*rms(out)/(a*(1+w)), alpha=1, d24=mean h24 last-prompt (let - heres, gen_1), coord 1038 zeroed, rms=.1151, a=2.494, 3 coords with tiny scale dropped. Prediction: heres answers flip to Let; flipped problems gain accuracy.

pass9 result: cand_let1 flips 120/125 heres->let (mechanism OK) but paired +0.8+-1.8; flipped subset +0.8+-3.6 (10 w2r, 9 r2w), cut .233->.35; natural let vs heres on c2 .168 vs .160 => mode gap NOT causal, did not replicate. Mode line CLOSED, edit dropped. Saved in cand_let1 (do not submit).

Failure tally 768 orig answers: acc .184, cut 27.5%, other wrong 38%, gold passed 6.4%, stopped at ref intermediate 5.3%, answered given number 4.4%.
pass10: TF original, no edit. texts=/work/tf10_texts.json: each cut answer truncated at loop start or after 2 cycles + exit string (Final Answer boxed / ####) + gold. Measures ceiling: share where every gold digit is greedy top-1.

pass10 result: exit ceiling natural format 2.9-4.3% of cut answers (any of 4 exits 5.7%) = 1.2-1.6pt of accuracy => loop exit CLOSED (loops are a symptom).
Arithmetic reparse with whole chains: 9.9% err overall; 2-term <=2d 1-2%; 3+d/decimal 11-17%; chains 34-62%; some residual parse artifacts. acc .25 with no wrong eq vs .01 with.

Chain sums: err 2 terms ~1%, 3 terms 32%, 4 terms 74%; 16/48 wrong = true minus one term. Wrong-eq answers where true value=gold: 16 (2.1pt), 10 of them chains. Perfect arithmetic ceiling est 3-4pt. Operator confusion on refs: no real bias (hyphen ambiguity); CLOSED. Stop errors balanced (over 6.4 vs under 5.3+4.4), no one-way bias.

Recency: first-digit copy accuracy on refs flat by distance (.85->.80), errors favour most recent number 42% but only 7% of events; CLOSED. Ref first-digit top1 ~83%, ~14.5 numbers per solution.

FEATURE SCREEN (768 orig answers, stratified by ref steps): time_units (minutes/hours) acc .045 vs exp .169, -14.5+-2.4 z-6 (n=110). week/day/year -9.9, per/each -8.2, weight/len units -7.9. fractions, money, more/less no deficit. LEAD, not mechanism yet.

Time deficit broken down: min only -11, hour only -16, both -11.7, ref uses 60 -12.7 vs no 60 -13.9 => NOT a 60-conversion failure. week -11.5 and per/each -7.8 persist without time problems. May be conceptual difficulty, not a defect; needs minimal-pair test (swap unit words in TF).

Time deficit is NOT in local number choice on refs (copy .838 vs .829; per-problem equal within steps) => failure is earlier, in deciding what to compute. Op share own vs ref suggests over-adding in rate/time groups but regex misses units; recounting.

Operator bias after corrected parsing and crediting 'half' etc: missing / 58 vs spurious 18 (one-way, p<.0001); + spurious 69 vs missing 39 (p=.005); -,* balanced. Finished needing / acc .184 vs .312 without; half of problems need /. Candidate failure: FORWARD APPLICATION (multiply/add givens when must invert). Partly circular; not yet causal.

Flip-fixable: 168 eqs in 136/627 wrong answers, 33 direct to gold (4.3pt, generous). used * need / 53 vs reverse 3 (one-way). But at the op token gap median 5.6 nats, need-op in top10 62% (35% for */), <1 nat 8% => op-token nudge CANNOT work, decision made earlier (quantity/role structure). Testing whether need-for-division is represented at last prompt.

Need-division signal at last prompt: d~.5 (null .1-.16), .9 within heres; neurons AUC .66-.69 stable. used vs skipped only d~.3. Weak, not clearly under-used; fixed divide-push would hurt no-div problems; no justified edit. Division line PARKED.
Next hypothesis: plain GSM8K reference style reasons better / avoids chain sums. Plan: steer via near-constant unit along mean(ref-own) on answer tokens.
pass11: TF original, no edit. texts=/work/tf11_texts.json = 240 own finished answers + 240 refs for the same problems (idx tf11_idx.json); hidden [6,12,18,24]. Purpose: mean(ref-own) on answer tokens.

pass11 result: mean(ref-own) on answer tokens split cos .98, paired d~3; axis = setup words/LaTeX (We, given, Let, $\) vs digits/=/space, i.e. setup vs compute. Write/out rms .17-.25 at alpha 1.
pass12: BOTH on dev_c3 (256). EDIT=cand_style11: model.layers.11.mlp.down_proj column 3585 += alpha*d12/(ka*(1+w)), alpha=1, d12=paired mean(ref-own) h12 answer tokens, coord 1038 zeroed, ka=mean(a/rms_out) on answer tokens, near-zero scales dropped. Writer active on prompt too. Prediction: less setup prose, more direct computation; judged on accuracy.

pass12 result: cand_style11 alpha=+1: paired -8.6+-2.3 (7 w2r, 29 r2w), cut .234->.230, tokens 317->298. Strong harm, significant. Hypothesis (ref style helps) refuted in this direction. Edit dropped; analysing how it harms.

How alpha=+1 harmed: drops restatement/definitions (We are given, so $A=60$, shorter headers), words 103->86, eqs 449->366, eq error 8.9->13.1%; reading proxy unchanged; only 4/29 lost answers changed mode. => restating quantities does useful work.
pass13: BOTH on dev_c3 (256). EDIT=cand_setup11_m1: same as cand_style11 with alpha=-1 (L11 down_proj col 3585 -= d12/(ka*(1+w))). Prediction: more setup/restatement, positive accuracy; watch cut rate.

pass13 result: cand_setup11_m1 alpha=-1 on c3: paired +0.8+-2.5 (22 w2r, 20 r2w), cut .234->.293. Style moved (words 101->120, eq 5.1->6.5) but both-finished +1.3+-3.5, only 5/20 losses from new cuts. Curve on c3: +1:-8.6, 0:0, -1:+0.8; rough fit peak near -0.6 ~+1.4. Step-split (+9.7 on 5+) is unplanned, noise-level.
NEXT IDEA: within-problem right-minus-wrong pairs from paired runs (~97 so far), balanced by which model gave the right answer.
pass14: BOTH on dev_c4 (202). EDIT=cand_setup11_m05: same writer alpha=-0.5.

pass14 result: cand_setup11_m05 alpha=-0.5 on dev_c4 (202): paired -5.0+-2.8 (11 w2r, 21 r2w), cut .208->.277, tokens 306->363. Curve: +1:-8.6 (c3), -0.5:-5.0 (c4), -1:+0.8 (c3). No reliable gain either sign => style-writer line CLOSED, edit dropped.

Loop/stop reanalysis (no pass): cut rate is flat by steps (1-step .34). Strict overrun (gold as eq result, not in problem): cut 24=1.96pt, finished wrong 29=2.37pt. Run-on numbered lists >10 items: 111 answers (9%), all cut, acc 2-3%; lists <=6 items cut <2%. Non-list cut .30 (exact loops). Ceiling if cut answers finished at same-progress accuracy 4.45pt. Stop gap to #### after gold 5.4 nats vs 7.9 legit; 5-nat shift flips 7/18 overruns, truncates 7/99 correct => stop threshold line CLOSED, no separable signal. Spurious constants: 5.5% of finished, own novel operand rate below refs => CLOSED. First-divergence right vs wrong: word vs word at token ~24, no class enrichment => style-level chaos, not the error site.

pass15: TF original, no edit. texts=/work/tf15_texts.json, pairs in tf15_meta.json: same problem answered right and wrong (paired runs), cut answers excluded, answers capped 420 tokens. hidden [6,12,18,24], mlp_act [11,17,23]. Purpose: within-problem right-wrong contrast balanced by which model was right.

Clock & repetition (no pass, pass7+pass15): residual has an answer-position CLOCK (hidden24 -380 to +250 over ordinary answers) and a separate REPETITION state: same position loop vs nonloop hidden12 clock-removed 243 vs 105, hidden18 167 vs -38. Written mainly by full-attn layers: L11 31% of u12, L17 37% of u18 (increment cos .52,.57). Coord 1125 is loud (written 4x, read weakly), AUC .67, not a clean detector. Dirs saved rep_dirs.npz. Position-degrade of arithmetic was a chain-tail parse artifact: small ops 0-2.6% flat => clock harmless so far.
pass16: TF original, no edit. texts=/work/tf16_texts.json [train_293 exact loop, train_24 list run-on]. attn_layers [5,11,17,23]. Purpose: which head reads the same phase of the previous cycle.

pass16 result: REPLAY HEADS. Mass on repeat sources (same 2-token context + same next token) at repeat count 1/2/3+: L5h2 .45/.60/.73 (run-on .39/.57/.88), top-attn on source 75-80%; L5h3 .41-.71; L11h1 .53-.67; L17h0,h2,h3 .35-.54; other heads near chance (.02). Prompt mass with source falls to .14 vs .80 without source (L5h2). In loops the model stops reading the problem and reads itself.
pass17: TF. EDIT: model.layers.5.self_attn.o_proj columns 512:768 (head 2) *0 (ablation). texts=/work/tf17_texts.json = tf7 (89 loops) + tf2 refs 512 + tf2 own first 423. Baselines pass7 (loops), pass2 (refs, own). Prediction: replay falls at cycle entry; ref digits little changed.

pass17 result: L5h2 ablation TF: loops any-token-leaves 68/62/43/31% by cycle (base 0 by construction), lp -.08->-.12, entry 5 tokens all stay 86.5%. Refs digits -.3 to -.8pt only. Own correct retention .981, wrong .971, cut .987. => contributes to replay, not sole, cheap for digits.
pass18: BOTH on dev_c0 (256). EDIT: model.layers.5.self_attn.o_proj columns 512:768 (head 2) *0. Prediction: cut falls; judge on accuracy.

pass18 result: L5h2 ablation on c0: paired -3.5+-2.1 (10 w2r, 19 r2w), cut .297->.355. Prediction REFUTED: attending the repeat source does not make it the loop driver; removal raises cut. It also reads the prompt on new text.
pass19: BOTH on dev_c1 (256). EDIT: model.layers.5.self_attn.o_proj columns 512:768 (head 2) *1.5. Prediction if gain is limiting: cut falls, accuracy up.

pass19 result: L5h2 x1.5 on c1: paired -0.4+-1.7 (9 w2r,10 r2w), cut .270->.246. Curve 0x -3.5/+5.9 cut, 1.5x flat. Head necessary, not limiting. Gain line closed for L5h2. Idea: only loops are a true gap between LM objective and greedy decoding.
pass20: BOTH on dev_c2 (256). EDIT: model.layers.17.self_attn.o_proj columns 512:768 (head 2) *0. Prediction if L17 replay drives loops: cut falls without accuracy loss.

pass20 result: L17h2 ablation on c2: paired -1.2+-2.0 (12 w2r,15 r2w), cut .270->.285, tokens 342->347. Prediction REFUTED: L17 replay head not the loop driver either. Edit dropped after this result.

Restart review (no pass): extraction rechecked on 1226 orig answers: gold in last line unextracted 0/385 => CLOSED. Marker acc .216 vs nomarker .315. Wider weight scan (zeros, kurtosis, gate/up cos, by layer): only mild kurtosis outliers L2 down_proj, L10 up_proj; no defect signature.
Pass15 right-wrong contrast finished: naive 23-40 neurons |t|>4 was artifact; balanced by which model right + cluster sign-flip per problem gives 1-3 neurons vs null 95pct 4-5. Split cos .25-.31 about 1 null sd. No reliable right/wrong direction => steering on it NOT justified, CLOSED.
Edits: none active (L17h2 dropped).

Reuse/ex analysis (no pass): numbers used as ref (ex=mi=0) acc .445; ex>0 acc .076, flat in steps; adjacent same-op replay rarer in own than ref => ex is comprehension, not eq replay. CLOSED.
X33 STATE: raw x33 prompt<pre<c1<later at L5,L11,L17. q col33 top norm at L4,L5,L11 (input_ln gain ~11). Largest single contributor to pre->later query shift L5 15-48pct, L11 19-78pct; L17/L23 are 350,367,775. q33.k33 cos L11 all heads .19-.37; L5 only h2 .36 (replay head, largest shift .45). Feedback not specific to replay heads overall. Hypothesis: x33 gains prompt-vs-self preference.
pass21: TF. EDIT cand_x33_b0: L5 and L11 q_proj all heads: col33*=0, col1038 += col33_orig*r (r=x33/x1038 after-norm pre-loop means, L5 -0.365, L11 -1.048; rel change .045,.081). Summary check: mean query kept; loop shift mostly down (L11h2 up .26->.31); prompt queries change .15-.39 rms = main risk. texts=/work/tf17_texts.json (loops+refs+own), return as pass17, baselines pass7, pass2. Prediction: loops leave cycle more, digits little changed. Digit loss on refs would refute usefulness.

pass21 result: cand_x33_b0 TF: loops any-token-leaves 36/11/6/0 pct (L5h2 ablation 68/62/43/31), lp -.12->-.13, entry all stay .978 (.865). Refs digits -.07 to -.17pt (pass17 control reproduced). Own retention .996/.993/.997. => x33 tracks state, does not sustain loops; 8pct tensor change nearly inert; harmless. But TF leave rate did NOT predict gen in pass17/18.
pass22: BOTH on dev_c0 (256). EDIT=cand_x33_b0 unchanged (L5,L11 q_proj col33 clamp beta 0). Prediction: cut within ~2pt, accuracy within noise; then x33 line closes.

pass22 result: cand_x33_b0 on c0: paired +1.95+-1.41 (9 w2r, 4 r2w), cut .297->.270, tokens 363->344. 94/256 identical, first divergence median tok 65, modes unchanged. Cut transitions 23 out,16 in; cut->finished correct 3/23. Both-finished +1.8+-1.8 (6 v 3). Not purely loop avoidance. Best so far but selected, z 1.4 => lead only.
PLAN: replicate SAME edit unchanged on c1,c2,c3, pool; only then dose (beta -1), then HOLDOUT.
pass23: BOTH on dev_c1 (256). EDIT=cand_x33_b0 unchanged. Prediction if real: accuracy positive, cut down a little.

pass23 result: cand_x33_b0 on c1: paired +1.17+-1.70 (11 w2r, 8 r2w), cut .270->.277, tokens 351->353. Cut drop NOT replicated; loop-avoidance claim dropped. Pooled c0+c1 about +1.6+-1.1.
pass24: BOTH on dev_c2 (256). EDIT=cand_x33_b0 unchanged.

pass24 result: cand_x33_b0 on c2: paired +0.39+-1.30 (6 w2r,5 r2w), cut .270->.293, tokens 342->354. Pooled c0-c2 768: 26 w2r,17 r2w, about +1.2+-.85, cut flat .279->.280. Estimates falling 1.95,1.17,.39 = selection-like.
pass25: BOTH on dev_c3 (256). EDIT=cand_x33_b0 unchanged. Last planned replication; pool after this.

pass25 result: cand_x33_b0 on c3: paired -0.78+-1.56 (7 w2r,9 r2w), cut .234->.231. POOLED c0-c3 1024 answers: 33 w2r, 26 r2w, +0.68+-0.75, cut 270->272. No demonstrated gain, no cut effect, first pass was selection. Not sent to HOLDOUT. Copy in cand_x33_b0; edit dropped. Baseline flips per 256 with near-inert edit about 12 v 11 = noise floor.

CLOSED no pass: time deficit not unit mismatch (factor rel 8pct all groups), final=last result (only 2/658 last=gold), marker extraction ok, number words no deficit. Time/rate gap real (-9.5,-7.3,per -3.5) = conceptual.
ATTENTION SINK: answer tokens put .19-.50 mass on BOS (pos0) in most heads, also pos3 newline at L17; problem text only .06-.41. Sink varies by next-token class (L23h3 .62 digit, .27 op; L5h2 sd .31) => gating/threshold, not constant. In loops replay heads leave sink AND problem (L5h2 .05/.03/.92 own); L23 keeps sink. BOS key: all energy low-RoPE dims, same direction as constant key part (cos .7-.88), 1.5x amplitude; mean-query gap bos-problem +4 to +12 tracks bos mass r .68. Hypothesis A: threshold too high, heads under-read problem. Counter B: filters noise, lowering hurts. q_norm dead dims 0-6 per layer, none in low dims.
pass26: TF on tf3_texts (2 texts) attn_layers 5,11,17,23, token_logprobs. EDIT cand_sink_d2: q_proj col1038 rows of heads L5 h0-3, L11 h0-1, L17 h0-3, L23 h0-3 += -(16*delta/|d|^2)*d*rq/((1+wq)*m), d=(k_bos-k_problem) low live dims only, delta=2, rq mean query rms, m mean x1038 after norm. rel change .4-.6pct. Prediction: BOS odds fall ~7x, other regions scale ~1.5 proportionally. Mechanism check only, not accuracy.

pass26 result: cand_sink_d2 mechanism OK: BOS odds down x3.2-5.7 (predicted 7), untouched L11h2,h3 ratio .9. Freed mass spreads proportionally incl own text (L5h0 own .33->.45, problem .17->.22), not aimed at problem. dlp answer -.02, digits 0/-.025, prompt tokens -.085/-.30 (prompt encoding changes).
pass27: BOS... BOTH on dev_c0 (256). EDIT=cand_sink_d2 unchanged (14 heads, delta 2). A: accuracy up. B: accuracy down. Cut may rise. Will test dose and opposite sign before believing it.

pass27 result: cand_sink_d2 on c0: paired +0.78+-2.35 (19 w2r,17 r2w), cut .297->.371, tokens 363->396, wrong_answer 126->108. No accuracy gain, cut up strongly. Neither A clean nor B clean; leans B (sink filters). Not replicated yet.

Open after notes reread: repetition state u12/u18 (rep_dirs.npz) written by L11/L17 was never tested causally (only head ablations, x33 clamp). Candidate: remove it from o_proj output.
pass28: BOTH on dev_c0 (256). EDIT=cand_sink_m2 = exact mirror of cand_sink_d2 (W_orig - (W_d2 - W_orig)), same 14 heads q_proj col1038, delta -2 (BOS odds UP). Predictions: B (sink filters noise): cut falls, accuracy flat/up. A (under-reading): accuracy down. Same problems as pass27 for a three-point curve.

pass28 result: cand_sink_m2 delta -2 on c0: paired -3.9+-2.2 (11 w2r,21 r2w), cut .297->.344, tokens 363->394. Curve delta -2/0/+2: acc -3.9/0/+.8, cut +4.7/0/+7.4. B refuted (raising sink does not cut loops); cut rises both ways => near a loop minimum, generic damage raises cut. Sink line CLOSED except possible low-dose +1. Edit dropped.

NEW LINE: LATE-EXIT CIRCUIT. Finished DEV answers past 400 tokens: 8/910, 0 right; 316/1226 run to 768; lucky cut right 7 (no marker, last number coincidence). So tokens past ~400 mark a lost answer. Edit active only there keeps normal answers identical => low-noise paired test. Rescue value: exit ceiling pass10 ~1.2-1.6pt; restart might do better (C2F rescued answers right 46/296=15.5pct, but those diverged earlier, so restart value is speculative).
Text-level within-problem R-W on 165 mixed problems: nothing passes Bonferroni (frac_cmd z 2.7, content not style). Style closed at text level. Problems: 69pct always wrong, 10 always right, 20.5 mixed.
pass29: TF original, NO edit (edited empty). texts=/work/tf29_texts.json, sel /work/tf29_sel.pkl: 30 cut non-list + 20 cut lists + 22 longest finished + 18 random finished. Return token_logprobs, topk 20, hidden 12,18,24,25,26, mlp_act 23,24,25. Purpose: detector for late regime, clock in loops, confidence at loop tokens, end-of-solution state.

Detector analysis (no pass, pass29 data): raw h18 direction d18 late-early: 94pct of cut answers past max of 680 control answers. But MLP reads h/rms: NORMALISED only 70pct past ctrl max (loops .57, lists .90), 90pct past ctrl q99; long finished 2/22. x1038/rms=32.8, CV<1pct = superb bias, usable at L18 only by raising pre_ff weight[1038] and zeroing col 1038 of gate/up for other neurons. Capacity: L18 16.9 nats, L24 46, L25 40; firing tokens need ~18-22 nats. Two-stage circuit L18 flag -> L25 writer, 7 tensors. Exit value: #### form gold top-1 5.7pct loopstart, 4.7 after2; lucky cut 2.2pct; net ceiling about +.7, with FP about +.4-.5. PARKED, not abandoned. Raw BOS-step at abs 512: none (d'.2-.3), refuted. Token rms at h18 CV .26-.3 (x1038 varies by token, ratio stable).
pass30: BOTH on dev_c1 (256). EDIT cand_rep_b1: oblique removal of late-state direction from o_proj output, L11 with dirs[(12,all)], L17 with dirs[(18,all)], beta 1. Dirs in late_dirs.npz. Prediction if state sustains loops: cut falls clearly, accuracy flat/up. If cut flat or up, state is a marker, line closed.

pass30 result: cand_rep_b1 on c1: paired +1.6+-2.2 (18 w2r,14 r2w), cut .270->.266, tokens 351->358. Cut flat, prediction not met at beta 1; note edit did NOT raise cut unlike other perturbations. Writers only ~1/3 each, so state likely survives.
pass31: BOTH on dev_c1 (256). EDIT cand_rep_b25: same direction, beta 2.5 (writers reversed) at L11 and L17 o_proj. Last test on this direction. Prediction if state sustains loops: cut falls with dose. If cut does not fall, state is a marker, line CLOSED.

pass31 result: cand_rep_b25 on c1: paired +2.7+-2.4 (22 w2r,15 r2w), cut .270->.285, tokens 351->389. Cut did NOT fall => repetition state does not sustain loops, loop-driver hypothesis CLOSED. Unplanned: both-finished tokens median +27 (b1 +6), both-finished acc +5.4+-3.1 (b1 0.0+-2.9), eq err 8.3->9.3, modes unchanged. Post hoc reading: direction has clock part, reversing writers slows clock, answers longer. LEAD only, c1 is discovery set.
RULE fixed before more data: replicate cand_rep_b25 UNCHANGED on fresh c2,c0,c3 (768), pool WITHOUT c1. HOLDOUT only if fresh pooled >= +2. Else close like x33.
pass32: BOTH on dev_c2 (256). EDIT=cand_rep_b25 (L11 o_proj dir (12,all), L17 o_proj dir (18,all), beta 2.5). Prediction: accuracy positive, both-finished answers longer by ~25 tokens.

pass32 result: cand_rep_b25 on c2 (fresh): paired +1.95+-2.38 (21 w2r,16 r2w), cut .270->.227, tokens 342->353. Cut moved opposite to c1, no loop claim.
pass33: BOTH on dev_c0 (256). EDIT=cand_rep_b25 unchanged. Fresh set 2 of 3.

pass33 result: cand_rep_b25 on c0 (fresh): paired +0.39+-2.63 (23 w2r,22 r2w), cut .297->.262, tokens 363->373. Estimates 2.7(disc),1.95,.39 falling, like x33. Fresh pool so far +1.17.
pass34: BOTH on dev_c3 (256). EDIT=cand_rep_b25 unchanged. Fresh set 3 of 3, then apply rule.

pass34 result: cand_rep_b25 on c3: paired 0.0+-2.6 (22 v 22), cut .234->.3125, tokens 317->387. FRESH POOL c2,c0,c3: 66 w2r,60 r2w, +.78+-1.46 < +2 => CLOSED per rule, not sent to HOLDOUT. All four +1.27+-1.25 (88 v 75). Length effect real (both-finished median +27,+33,+27), accuracy not. Cut changes disagree (+1.6,-4.3,-3.5,+7.8). Copy in cand_rep_b25.

pass35: TF edited. EDIT cand_probe18 (read-only, writes nothing): L18 pre_ff weight[1038]=-1+2^-6; gate_proj and up_proj col 1038 zeroed except probes; probes n1408 reads dn18 (dn18.npy), n5843 bias only, n3144 u18, n6532 v18; each gate col1038 = 8/(eps*32.8), up col1038 = 1/(eps*32.8), down columns zero. So a=gelu(read+8)*bias. texts=/work/tf35_texts.json, sel tf35_sel.pkl: 90 tf29 + all right finished DEV + wrong finished, 390. Return mlp_act 18, token_logprobs. Checks: bias near 1; cut answers exceed right answers; lp close to pass29.

pass35 result: probes on real L18 input. Bias mean 1.002 CV .8pct. State per-answer max: right finished (252) q99 2.13 max 2.33; all finished ctrl 318 none past 2.4; cut fire 70pct at 2.4 (lists .90 loops .57), 80pct at 2.2 (FP right .8pct), 98pct at 2.0 (FP 2pct). Hijack not inert: lp diff q99 .30. VALUE: 25.8 cut x .7 fire x (5.7 exit - 2.2 lucky) = +.25 to +.4pt, unprovable. EXIT CIRCUIT CLOSED as candidate; detector itself valid. Edits dropped.

CHAIN LINE (no pass so far): strict parse DEV 1226 orig: chain ok n=2 .93, n=3 .58, n=4 .13, n>=5 0; 227 answers with chain, 119 with wrong chain (acc .059); direct gold if chain right 17 answers=1.39pt. Wrong chains: same length, tens/hundreds digit, true-gen=+1 41/59 (carry), plus true minus one term. Idea: split chains into running sums (a+b=S + c), 2-term adder nearly perfect. Gap lp(+)-lp(=) at operand-2 end when continuing: median 8.2 (q10 4), operand-1 9.9 => decisive plan, gate must separate operand 2 from 1. Single neuron selectivity (pass29 scouting, rms bystanders) max L23 n2099 2.6, L24 2.8, L25 2.0. L25 write only affects same-token logits.
pass36: TF original, NO edit (edited empty). texts=/work/tf36_texts.json, sel /work/tf36_sel.pkl: 200 DEV own chain answers cut to 420 tokens (group A) then 200 same answers cut at operand 2 + ' = S' true partial sum (group B). Return token_logprobs, topk 20, hidden 25,26, mlp_act 24,25. Prediction if splitting viable: ' +' top-1 after S in most of B. If '.'/'$'/newline dominates, model drops remaining terms, line CLOSED.

pass36 result: chain split REFUTED by pre-set criterion: after forced ' = S', ' +' top-1 only .5pct, lp(+) median -7.0, closing token top with lp -.01 ('$.' 105/200). Model treats = as end of expression, splitting drops remaining terms. CHAIN LINE CLOSED, no gen pass spent. Carry-digit nudge not pursued: needs digit-specific gating, ceiling under .5pt.
NEW LINE: PROBLEM-READING STEER. Never tested directly: heads biased toward problem tokens vs own text (sink edit freed mass that went mostly to own text). Method as sink: constant query offset via q_proj col1038, d=k_problem-k_own low live dims.

Read steer build: per layer L, v1=(k_problem-k_own)*live, v2=(k_bos-k_own)*live, u=v1-(v1.v2/v2.v2)v2, D1=u/(u.v1). So problem vs own shift +delta, BOS vs own 0 (plain p-o would raise BOS 1.1-1.6x, i.e. sink, which hurt). q_proj col1038 rows of head h += 16*delta*D1*rq_h/((1+wq)*m_L), live dims as cand_sink_d2 (RoPE slow dims i>=~58). m_L 6.08,4.47,5.33,26.68 recovered from cand_sink_d2 (rows agree).
New comparator digit_table(pass,sel,base): ref digits by kind copy/reuse/result/other x first/later, cluster se. Validated: pass21 -.13, pass17 -.55 overall. Base first digits copy .774 reuse .871 result .932 other .591.
pass37: TF edited. EDIT cand_read_p1: delta=+1, all 16 heads L5,11,17,23 q_proj col1038, rel change .55-.92pct. texts=/work/tf17_texts.json, return token_logprobs topk 10. Baselines pass2 (refs, own), pass7 (loops). Prediction if problem reading limiting: copy first up >=1pt, reuse first not down comparably. Flat/negative refutes; copy up paid by equal reuse loss = mass shift only.

pass37 result: cand_read_p1 TF refs digits: copy first +1.18+-.39, copy later +.73, other first +1.50, BUT reuse first -2.05+-.38, result first -1.41+-.35; ALL -.22. By pre-set rule = MASS SHIFT, not comprehension gain. Answer dlp refs +.064 looked huge but is ONE TOKEN: #### at ref end -7.54 -> -3.16 (2243 of 2442 total gain); excluding top 3 tokens dlp -.0002. Ops slightly worse. So NOT better reasoning. Top1 all tokens .7335->.7341. Prompt dlp -.067.
Unplanned: edit raises prompt-only marker retrieval by 4.4 nats at end of solution, first non-gated edit moving stop machinery. Could cut loops or truncate; unknown net.
pass38: BOTH on dev_c0 (256). EDIT=cand_read_p1 unchanged (delta +1, 16 heads). Prediction: cut below .297 clearly, tokens below 363. Accuracy open; judge on paired accuracy only, fewer cuts alone is no gain.

pass38 result: cand_read_p1 on c0: paired +1.17+-2.18 (17 w2r,14 r2w), cut .297->.125, tokens 363->262, marker share .24->.45, wrong_answer 126->167. FIRST edit to halve cuts, but cut->finished 54 answers only 1 right (1.9pct) => LOOPS ARE A SYMPTOM, confirmed causally; cut drop is not a gain. Both finished 170: .312->.329 (16 v 13). r2w 12/14 never reach gold, length unchanged => losses not premature stops, more likely reuse/result cost.
Next idea: NUMBER STEER: steer heads to digit tokens anywhere (problem+own), BOS and problem-vs-own unchanged. Sharp prediction copy first AND reuse first both rise. Edit kept in cand_read_p1.

Number steer build: KT[L] class mean keys from pass7 (89 texts; validated vs SK cos .97-1.0). vd=.5*((pdig-pword)+(odig-oword))*live; orthogonal to c1=(bos-word)*live and c2=(problem-own)*live via QR of those 2 summary vectors; D1=u/(u.vd). cos pdig-word vs odig-word .74-.89, kept .92-.97, shifts problem digit 1.04-1.11, own digit .97-1.06, BOS 0. Dirs in steer_dirs.pkl.
pass39: TF edited. EDIT cand_num_p1: delta=+1 number steer, all 16 heads L5,11,17,23 q_proj col1038. texts=/work/tf17_texts.json, topk 10. Criterion set now: copy first AND reuse first both >= +.5, result first not below -.5. Else mass transfer, close without gen.

pass39 result: cand_num_p1 TF refs: ALL digits +.51+-.13 (best of session), copy first +1.46+-.38, copy later +.67, result first +.45, result later +1.32+-.31, reuse first -.95+-.32, reuse later +.12, other first -.27. op/eq top1 +.85, not_in_problem words -.54, #### -.13 nats (read steer +4.4), prompt dlp -.007. PRE-SET CRITERION FAILED on reuse first (needed >=+.5). OVERRIDE recorded: continuing because net digits positive z~4, unlike read steer -.22, so not plain transfer. This is a post hoc decision, discount accordingly. DEV only.
pass40: TF edited. EDIT cand_num_p2: number steer delta=+2, same 16 heads. texts=/work/tf17_texts.json, topk 10. Dose test. Prediction: net digits ~ +.8 or more, reuse first worse. If net falls vs delta 1, stay at delta 1.

pass40 result: cand_num_p2 delta 2 TF refs: ALL digits +.74+-.17 (delta1 +.51), copy first +2.60, result first +.85, result later +1.62, reuse first -1.57, other first -2.18; words not_in_problem -1.44 (d1 -.54), in_problem -.54 (d1 -.02); ALL TOKENS top1 +.127pt at d1, -.176pt at d2. Monotone dose = causal. Digit gain saturates, word cost superlinear => STAY AT DELTA 1.
PLAN FIXED BEFORE ANY GEN OF THIS EDIT: cand_num_p1 UNCHANGED BOTH on c0,c1,c2,c3, pool. Freeze best DEV candidate, measure on all 4 HOLDOUT sets. SUBMIT edit only if pooled HOLDOUT gain > 0, reporting SE even if not significant; else original.
pass41: BOTH on dev_c0 (256). EDIT=cand_num_p1 (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23). Prediction: paired accuracy positive, cut NOT halved (no #### effect).

pass41 result: cand_num_p1 on c0: paired +.78+-1.92 (13 w2r,11 r2w), cut .297->.273, tokens 363->350, wrong_answer 126->131. Cut NOT halved, as predicted. Splits {'orig_fin>ed_fin': (163, 53, 51), 'orig_fin>ed_cut': (17, 1, 0), 'orig_cut>ed_fin': (23, 0, 4), 'orig_cut>ed_cut': (53, 0, 1)}, identical texts 39, both-finished median dlen 0.0.
pass42: BOTH on dev_c1 (256). EDIT=cand_num_p1 UNCHANGED (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23). Pre-fixed pool c0,c1,c2,c3, no retuning between passes.

pass42 result: cand_num_p1 on c1: paired +.39+-2.11 (15 w2r,14 r2w), cut .270->.273, tokens 351->354. Splits (orig>ed, F finished C cut; n, orig right, ed right) {'FF': (158, 40, 42), 'FC': (29, 4, 0), 'CF': (28, 1, 4), 'CC': (41, 0, 0)}, identical 28, both-finished median dlen 0.0. Pool c0,c1: 28 v 25.
pass43: BOTH on dev_c2 (256). EDIT=cand_num_p1 UNCHANGED (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23).

pass43 result: cand_num_p1 on c2: paired +4.69+-2.12 (21 w2r,9 r2w), cut .270->.254, tokens 342->336. Splits {'FF': (161, 40, 44), 'FC': (26, 0, 0), 'CF': (30, 1, 6), 'CC': (39, 1, 4)}, identical 28, both-finished median dlen 0.0. Pool c0,c1,c2: 49 v 34, 1.95+-1.19. One strong set among three, do not overread.
pass44: BOTH on dev_c3 (256). EDIT=cand_num_p1 UNCHANGED (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23). Last DEV set of pre-fixed pool.

pass44 result: cand_num_p1 on c3: paired +1.95+-2.18 (18 w2r,13 r2w), cut .234->.203, tokens 317->302. Splits {'FF': (175, 62, 64), 'FC': (21, 3, 0), 'CF': (29, 1, 7), 'CC': (31, 0, 0)}. DEV POOL c0-c3: 67 v 47 of 1024, 1.95+-1.04. Splits pool {'FF': [657, 195, 201], 'FC': [93, 8, 0], 'CF': [110, 3, 21], 'CC': [164, 1, 5]}. All four sets positive (+.78,+.39,+4.69,+1.95). Winner's curse applies: many candidates tried on DEV.

FREEZE: cand_num_p1 frozen as best DEV candidate (copy /work/best_1, local hash 2f369a8c3d3b, tool edit_sha256 9e7bb2b8). DEV pool +1.95+-1.04 (67 v 47 of 1024), vs x33 +.68+-.75, rep_b25 fresh +.78+-1.46, read_p1 +1.17+-2.18 c0 only. Caveats: pool would just miss the +2 gate that was specific to rep_b25; this edit's pre-fixed plan had no threshold. Gain mostly from CF 110 answers 3->21 right, offset by FC 93 answers 8->0; FF 657 195->201. Winner's curse expected.
HOLDOUT RULE: run hold_0,1,2,3 ALL, unchanged, no stopping or retuning on interim results, judge pooled only. Submit edit iff pooled HOLDOUT gain > 0, report with SE even if not significant; else drop all edits and submit original. Spare passes after HOLDOUT cannot change the edit.
pass45: BOTH on hold_0 (256). EDIT=cand_num_p1 frozen (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23).

pass45 result: FROZEN cand_num_p1 on hold_0: paired +1.56+-2.14 (17 w2r,13 r2w), cut .328->.250, tokens 383->342, wrong_answer 126->143, sha 9e7bb2b8 same. No decision on one set.
pass46: BOTH on hold_1 (256). EDIT=cand_num_p1 frozen, unchanged.

pass46 result: FROZEN cand_num_p1 on hold_1: paired +.78+-2.07 (15 w2r,13 r2w), cut .258->.305, tokens 340->367, wrong_answer 144->130, sha same. Interim hold_0,1: 32 v 26 of 512, about +1.17+-1.49. Cut direction flips between sets, no consistent cut effect. No decision until all 4.
pass47: BOTH on hold_2 (256). EDIT=cand_num_p1 frozen, unchanged.

pass47 result: FROZEN cand_num_p1 on hold_2: paired +.39+-2.38 (19 w2r,18 r2w), cut .270->.242, tokens 344->336, wrong_answer 131->134, sha same. Interim hold_0-2: 51 v 44 of 768, about +.91+-1.27, below DEV as expected. No decision until all 4.
pass48: BOTH on hold_3 (256). EDIT=cand_num_p1 frozen, unchanged. Last HOLDOUT set.

pass48 result: FROZEN cand_num_p1 on hold_3: paired +.78+-1.84 (12 w2r,10 r2w), cut .258->.266, tokens 345->351, sha same.
HOLDOUT POOL hold_0-3: 63 v 54 of 1024, orig 20.31 edit 21.19, gain 0.88+-1.06, exact paired p 0.460. Positive, NOT significant. Under half of DEV +1.95, as selection predicts.
DECISION BY PRE-SET RULE: gain>0 => SUBMIT cand_num_p1. Final. Last 2 passes cannot change it. No split analysis of HOLDOUT generations (said so during the run), so rescue pattern stays DEV-only.

Comparator control OK: digit_table(39,sel,base,kbase=2) reproduces logged table. Ref format = PROMPT + solution minus <<..>> stripped + newline + '#### ' + answer.
SPARE PASSES (cannot change submission): HOLDOUT mechanism test. texts=/work/tf49_texts.json (all 1024 HOLDOUT refs, order holdout_ids), token_logprobs, topk 10. Compare with digit_table(50,range(1024),range(1024),kbase=49), T_cur=T_h.
PRE-SET: mechanism CONFIRMED if net digit top1 >= +.25 and z>2, with copy first >0 and reuse first <0. REFUTED if net near 0 (z<1). In between = partial. Either way submission stays cand_num_p1.
pass49: TF ORIGINAL on /work/tf49_texts.json. Edit on disk unchanged (cand_num_p1), not used by this pass.
pass50: TF EDITED, same file, cand_num_p1 frozen.

pass50 result (HOLDOUT 1024 refs, edit vs original, digit_table): net digit top1 0.693+-0.084 z 8.2 (DEV .510+-.122), copy first 1.742, reuse first -0.024 (DEV -.951), all answer tokens 0.161+-0.046 top1 0.7338->0.7354, marker dlp mean -0.133 median -0.101. Table [['copy', 'first', 5108, 0.772, 1.742, 0.282, 0.05], ['copy', 'later', 3780, 0.958, 0.899, 0.175, 0.027], ['other', 'first', 1383, 0.651, -0.651, 0.474, -0.063], ['other', 'later', 1015, 0.776, 1.084, 0.646, 0.021], ['result', 'first', 3469, 0.915, 0.432, 0.202, 0.027], ['result', 'later', 4437, 0.948, 1.082, 0.196, 0.031], ['reuse', 'first', 4134, 0.859, -0.024, 0.25, 0.012], ['reuse', 'later', 4972, 0.976, 0.181, 0.105, 0.009]].
VERDICT: digit mechanism CONFIRMED on HOLDOUT. Reuse clause passed by sign only, reuse first is flat, DEV cost did NOT replicate, so do not claim it as replicated. First run of this script crashed in the descriptive part (double bos in my lengths), digit_table unaffected.
All 50 passes used. SUBMISSION cand_num_p1 in /work/edited, identical to best_1: True. HOLDOUT accuracy +.88+-1.06, p .46, not significant.
