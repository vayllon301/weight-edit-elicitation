# Notes
DEV = /work/dev_ids.json (1250), HOLDOUT = /work/holdout_ids.json (1000), seed 20261107; never tune on HOLDOUT.
dev_a=DEV[:512], dev_b=DEV[512:1024].
Pass 1 plan: original on dev_a, generation with activations. No edits yet.

Findings so far (pass 1, original on dev_a, 512): acc 18.2%, cut 27.7%. All answers open 'Let' or 'Here'. Let: acc 20.8, cut 33.5, acc|fin 30.2. Here: acc 14.9, cut 20.6, acc|fin 17.7. Stratified by steps Let-Here +7.4 pts.
Loops: 120/138 cut have exact repeated line; develops during generation, not predictable at prompt.
Near-constant last-prompt neurons exist, eg L18 n2543, L16 n180, L21 n5695, L10 n5796.
Pass 2 plan: teacher_forced ORIGINAL (no edits) on 220 own answers (90 cut,65 corr,65 wrong; trim 270 answer tokens) /work/tf2_texts.json, meta tf2_meta.json, hidden 11,14,17,19,24,26, mlp 13,18,23, topk 10.

BLOCKED twice: token-level selectivity checks of the planned write neuron. Do not retry; use pass-1 population screen only and learn effects via forward passes.
Pass 2 done: TF top1 matches greedy 99.65%. First token is 2-way Let/Here; here-mode margin median -1.25, let-mode +1.62.
EDIT E1 (pass 3): L18 down_proj column 2543 += dy/abar, dy=alpha*T*yrms/(1+w); T=Let-Here mean last-prompt hidden[19] (pass1 dev_a); alpha=1.5. Problem set dev_s1=DEV[512:768]. Hypothesis: flips Here->Let, accuracy rises.

Pass 3 result E1 (mode flip Here->Let, L18 n2543 alpha 1.5): flipped 108/108 Here to Let on dev_s1; acc -0.4+-1.8; flipped group acc 21.3->19.4, cut 20.4->29.6, acc|fin 26.7->27.6. MODE LINE CLOSED: mode is selection not cause. Edit dropped, copy in /work/cands/e1.

Sizing: chain errors in 25% of failures but true value==gold only 16/615. Loop onset top1 median .78, P(finishing word) median .007.
Pass 4 plan: teacher_forced ORIGINAL, no edits. /work/tf4_texts.json: 800 DEV reference solutions + loop-break probes (append final-answer line with gold at loop onset). topk 5. Purpose: token-class first-break analysis and loop-stop payoff bound.

Pass 4: loop-stop probes greedy gold only 6/184 (3.3%): LOOP-STOP LINE CLOSED. Reference prefixes: result digits top correct .92-.94; first break: fromproblem 357, othernum 173, fromearlier 96, result 71, none 103 of 800. Main mechanism candidate: retrieving next operand from the problem.

Retrieval breakdown: fresh/reused and early/late all ~.73-.82, wrong digit mix fresh 291 used 211 none 292: no simple bias.
Pass 5 plan: teacher_forced ORIGINAL (no edits), two multi-turn texts (tf5_texts.json) of DEV problems + reference solutions, attn_layers 5,11,17,23: find retrieval heads.

Pass 5: attention at full layers. L17 head0 = retrieval head (src .21, other digits .178, peak on source 39.7%, ctrl digits .035). Others weaker: L5h2, L17h3, L11h1.
EDIT E2 (pass 6): L17 o_proj columns 0..255 (head 0) x1.5. TF on tf4_texts (984) for paired token-class comparison vs pass 4.

Pass 6 E2 (L17 h0 x1.5, TF refs): fromproblem_first -0.13+-0.17 pts, result_first -0.49+-0.20. BOOST DOES NOT HELP; head correlates with success (peak 47.7% vs 14.7%) but not limited by gain. E2 dropped.

Attention at result digits: L17 h2,h3 (and h1 first digit) read operands (hit .76-.85). L17 h0 retrieves from problem. Unique equation error: 2-term small + 11%, 3-term + 25-70%, 4-term + 79%, * small 3%, * big 16%.
Pass 7 plan: ORIGINAL TF baseline on tf7_texts.json (450 refs + 574 arithmetic contexts with corrected results), logprobs/topk5.

Pass 7: tf7 baseline. Whole-result acc given correct prefix: + 2 small .974, big .879; + 3 small .815, big .393; + 4 small .227; mixed .30; * .97-.985. Multi-term sums are a specific failure; 30/66 wrong multi-term sums are partial sums, 22 drop the last term.
EDIT E3 (pass 8): L17 o_proj columns 512..1023 (heads 2,3) x1.5. TF tf7 vs pass 7.

Pass 8 E3 result (L17 h2,h3 x1.5, TF tf7 vs pass 7): whole-result all +0.05+-0.22; +3 +0.83+-1.86 (dlp +.014+-.005); +4 +7.69+-7.54 n=26; word acc +0.21+-0.06; fromproblem_later -0.44+-0.17; fromearlier_first -0.45+-0.19. Small logprob gain, no accuracy gain. Copy in cands/e3, edit dropped, no active edits.
Next: free text analyses on gen_1 + gen_3_original (premature finish, number coverage), weight anomaly scan. No pass.

Free analyses: wrong_fin fewer eqs than ref 28% (correct 5%); wrong miss >=1 needed number 26.6% (late numbers dropped 2x early); pred=ref intermediate 10%; gold in body 13%. Weight scan: no broken component. Failure is broad reasoning.
Pass 9 plan: teacher_forced ORIGINAL, no edits, /work/tf9_texts.json (meta tf9_meta.json): the 450 pass 7 refs under two contexts, 3 worked DEV examples (shots) and an expert instruction (care). logprobs, topk 5. Same tokens, compare with pass 7 by class. Purpose: unconfounded within-problem context contrast. If number tokens improve, get hidden states for direction.

Pass 9 analysis note: first table in run 33 mixed metrics (top-1 for contexts, top-digit for baseline); use run 34 stats_at only. dlp columns were valid: word shots +.112, care +.039; fromproblem_first shots +.007+-.009, care +.020+-.004; result_first shots -.040+-.006.

Pass 9 result (run 34 metrics): context priming flat on numbers: fromproblem_first shots +0.13+-0.49, care +0.22+-0.30; word shots +1.01+-0.12; all-numbers-right base .147, shots .131, care .133. CONTEXT-STATE STEERING LINE CLOSED: examples teach style, not math.
Mode line reopened in reverse only: E1 raised cut 20.4->29.6 on the same problems with acc|fin unchanged, reverse never tested.
EDIT E4 (pass 10): L18 down_proj column 2543 = original - (2.0/1.5)*E1 delta (E1 reversed, alpha -2.0). Set dev_s1=DEV[512:768], generate both. Hypothesis: Let->Here reduces cut with acc|fin unchanged, so accuracy rises. Copy in /work/cands/e4.

Pass 10 E4 (reverse flip, alpha -2) on dev_s1: acc 23.4->17.2, -6.25+-2.18, right->wrong 24, wrong->right 8, cut 25.8->25.0. HARMFUL, dropped. No active edits.

Pass 10 split: E4 flipped 103 Let->Here: acc 24.3->10.7, acc|fin 32.9->12.8, cut 29.1->24.3; not flipped 22.9->21.6. MODE IS CAUSAL ON LET-NATURAL PROBLEMS. E1 on Here-natural: acc|fin 26.7->27.6, cut 20.4->29.6. Let mode cut ~30% on both groups, Here ~21-24%. Next: read Let loops and Let features (free).

Run 40 proposal was refused by review (no reason given); it restored a helper through exec. Helper now written out in plain code, no exec. Name clash: flipped ids are fl_ids, fl is the number parser.

Mediators on same problems: E4 pairs Let .35 vs Here .13; false eq .145/.136, missing numbers equal, short equal, coverage .79/.64. E1 control: coverage .49/.71 with accuracy equal, so restating is not sufficient. No mediator qualifies.
First-line question overlap, within-steps +9.1+-3.5 (correlational). Goal mention: P(want|first We) .01, within-steps -1.5+-4.7: GOAL-STATEMENT LINE CLOSED. Wrong finals = last equation ~70% both registers, gold derived 10-16%: minor.
Loops: gold before onset 16% (Let), last eq=gold 1.4%: rescue by stopping closed again.
EDIT E3b (pass 11): L17 o_proj columns 512..1023 (h2,h3) x2.5. TF /work/tf7_texts.json, logprobs, topk 5, compare to pass 7 and pass 8 (x1.5). Dose test.

Pass 11 E3b x2.5 TF: word dlp reverses (+.0029 at x1.5, -.0021 at x2.5), fromproblem_later dlp -.028, whole results all +0.21+-0.27, +3 +4.17+-2.17, +2 -0.91+-0.48. Trade-off peaking near x1.5: OPERAND DOSE LINE CLOSED, E3 x1.5 still owed one generation pass. E2/E3 are mirror images along head share. No active edit.

First-variable binding: final=first var value in 12% of wrong Let answers, 3 wrong low-overlap cases: BINDING HYPOTHESIS CLOSED. Two-step problems acc 33%, errors mixed: false arithmetic, inverted rates, senseless plans.
EDIT E3 (pass 12): L17 o_proj columns 512..1023 (h2,h3) x1.5, only active edit. Generate both on dev_s2=DEV[768:1024], fresh for generation.

Pass 12 E3 x1.5 generation on dev_s2: 22.3->22.3, 0.0+-2.14, 15/15, cut 24.6->27.7. OPERAND REBALANCING CLOSED. No active edit. dev_s2 baseline 22.3.

Operator confusion on refs balanced (recall .75-.87, mul->div 81, div->mul 45): OPERATOR BIAS LINE CLOSED.
Pass 13 plan: teacher_forced ORIGINAL, no edits, prompts of DEV[:1024] (/work/tf13_texts.json, meta tf13_meta.json), topk 5. Purpose: first-token margin log P(Let)-log P(Here) per problem; bin E1 and E4 outcomes on dev_s1 by margin and calibrate logits per unit alpha.

Pass 13 margins (sign agrees with opener .974). Natural acc U-shaped: m<-2.5 30.3, [-1.5,-.75) 12.8, [-.75,0) 15.4, [0,.75) 18.7, then ~25. dev_s1 causal bins: E1 m<-2 -22.2+-8.0 (0/6), [-2,-1) +5.3+-5.2 (3/1), [-1,0) +4.7+-6.5 (5/3); E4 m in 0..2 -11 to -12. Crossover near m=-2 (POST HOC, must confirm elsewhere). Calibration asymmetric: E1 alpha 1.5 shift > 4.62, E4 alpha -2 shift ~2.6.
EDIT E5 (pass 14): L18 down_proj column 2543 = original + (0.6/1.5)*E1 delta (alpha +0.6). TF prompts tf13 vs pass 13 to measure the logit shift. Target shift +2. Copy /work/cands/e5_06.

Pass 14 E5 alpha +0.6 TF: margin shift +2.03 at m<-2, +1.92 at [-2,-1), +1.53 at m>2, std .25; boundary near m=-1.9; [-2,-1) 87% Let, m<-2 4.7% Let, no Let->Here flips.
Pass 15 plan: E5 alpha 0.6 active (L18 down_proj col 2543), generate both on dev_s2 (fresh for this hypothesis; dev_s1 excluded because it suggested the crossover). Prediction: flipped problems gain, about +1.5 to +2 overall.

Pass 15 E5 alpha .6 on dev_s2: 22.3->25.4, +3.12+-1.99, 17/9, cut 24.6->25.4. First positive generation result, fresh problems. Copy /work/best_1.
Sets for more paired passes: dev_s3=DEV[1024:] (226), dev_a1, dev_a2. Keep E5 active for pass 16 on dev_s3.

Passes 16,17 E5 alpha .6: dev_a1 +1.17+-2.11 (16/13), dev_a2 +0.78+-1.66 (10/8). Pooled with dev_s2 about +1.69+-1.11 on 768. Same edit active.

Pooled E5 on dev_s2+dev_a1+dev_a2 (768): +1.69+-1.11, 43/30, cut 26.7->28.8. Flipped n258 13.2->19.0 +5.81+-2.85 (35/20); kept -0.39+-0.83; no Let->Here. Boundary flipped bin m<-1.5 -4.26+-4.21 (n47), noise, strength stays +0.6.
Pass 18 plan: E5 alpha .6 active, generate both on dev_s3=DEV[1024:] (226) to finish DEV.
Idea reopened: third opener (register causal, E4 swing 20 pts acc|fin). Test hand-picked openers via a prompt-end-only write. No HOLDOUT yet.

Pass 18 E5 on dev_s3: +1.77+-2.17 (14/10). All four DEV passes positive; pooled 994: net +17 answers, about +1.7+-1.0. best_1 holds E5 alpha .6.

EDIT E6 We-opener (pass 19): L18 down_proj column 2543 += dy/abar, dy=alpha*T_we*yrms/(1+w), T_we = mean hidden_19 at newlines followed by We minus other newlines (pass 2 own answers, original), alpha=1.207 (|alpha T_we| = 3|T_lh|). TF prompts tf13, topk 5: measure whether We becomes top first token. Copy /work/cands/e6_we. E5 stays in best_1.

Pass 19 E6 TF: We top on 0/1024, Let 1006; We +5.3 lp, Let-Here +2.8. T_we splits: We vs mean(Let,Here) +5.9, Let-Here +2.8. Post-ff RMS caps a single-neuron write near the update norm (~2080 at L18), so scaling T_we is not viable. cos(T_we, We output direction) .18. Direct-path scale (|e|/rms) accounts for E1 and E6 shifts within ~15%: writes act by passing straight through to the output. NOTE name clash: w overwritten in run 59, use w18; E() is the encoder helper, embedding rows are in rows{}.
EDIT E7 We-opener output-side (pass 20): L18 down_proj column 2543 (from ORIGINAL) += dy/abar, dy=Delta*u*yrms/(1+w18), u=unit (1+wf)*(row We - mean(row Let,row Here)), wf=model.norm.weight, Delta=430 hidden units (hand target ~13 logits). No E5 in this edit. TF /work/tf13_texts.json, topk 5, compare pass 13. Copy /work/cands/e7_we. Next if We wins: generate both on dev_a1. E5 stays in best_1.

E7 as first written NOT RUN (|dy|/|ymean| .785, two low-gain dims held 42%). REPLACED by EDIT E7b (pass 20): L18 down_proj column 2543 (from ORIGINAL) += dy/abar, dy=S*unit(gp), gp=g minus its projection on glh, g=(1+w18)*(1+wf)*(row We - mean(row Let,row Here)), glh=(1+w18)*(1+wf)*(row Let-row Here), S=target*yrms*rms_f/|gp|, target=13 logits, rms_f=428.5, S=1.3979. Rows/norm weights read only. TF /work/tf13_texts.json, topk 5, compare pass 13. Copy /work/cands/e7_we. Next if We wins: generate both on dev_a1. E5 stays in best_1.

Pass 20 E7b TF: We top 0/1024, Let 724, Here 300; We lp -3.37, gap to best natural -3.10 (5%..95% -4.88..-1.64); Let-Here drift +.98+-.29; The +2.3, First +1.9, Okay +1.4. Actual shift >=7 vs target 13. Correction: direct path explains ~55-80% of E1/E6, not 'within 15%' (rms_f is 428, not 276). Leak: only token 3 back (newline after end of turn) +.675 dlp, others ~0.
EDIT E7c (pass 21): L18 down_proj column 2543 (from ORIGINAL) += dy2/abar, dy2=S2*(unit(gp)-kappa*unit(glh)), S2=2.2*S0=3.0755, kappa=0.161. Generate both on /work/dev_a1.json. Read: share of We openers, accuracy of We answers vs original and vs E5 on same problems (pass 16). Copy /work/cands/e7c_we. E5 stays in best_1. Next openers if the register has merit: First, The, To, Step, same recipe.

Pass 21 E7c on dev_a1: 22.7->21.5, -1.17+-2.57 (20/23), cut 27.0->32.0. 241/256 open 'We are given' (write works as a switch). We group acc: We 22.4, orig 23.2, E5 24.9. Natural Let 32.3->25.0, natural Here 13.7->19.7. Register rank Let>=We>Here; We is not better than Let overall.

E7c analysis: natural-Here flipped by E5: We 21.0 vs Let 19.8 (9/8); m>1 We -8; We adds nothing to E5. WE OPENER CLOSED as candidate. Tool kept: output-side write at n2543 selects any first token (S2=2.2*S0, kappa .161 for We: 94% We). No active edits.
Pass 22 plan: teacher_forced ORIGINAL, no edits, /work/tf22_texts.json (meta tf22_meta.json): first 102 tf13 prompts + each of 10 openers (First,To,Step,The,Okay,Solution,**,##,1,Given), token_logprobs + topk 3. Purpose: exact first-token logprob and continuation of each opener, to pick reachable openers.

Pass 22 screen done (first-token logprobs of 10 openers on 102 prompts), table in run output.

Screen: gap to best natural Okay -8.0, The -10.4, 1 -10.6, ** -11.0, First -11.2, To -12.6, Given -13.8, Step -14.2, Solution -16.6, ## -18.3. Continuations: First/Okay ',', To ' find', 1 '.', ** '1'/'Step', The ' problem'/' total'.
EDIT E8 First-opener (pass 23): L18 down_proj column 2543 (from ORIGINAL) += d/abar, d=2.4*Sb*(unit(gxp)-kap*unit(glh)), gxp=gain*(row First-base) minus projection on glh, Sb=13*yrms*rms_f/|gxp|=1.1780, kap=0.191. Generate both on /work/dev_a1.json. Bar: beat E5 23.9 on this set by about 3 points. Copy /work/cands/e8_first. E5 stays in best_1. Queue: Okay, The, To.

Pass 23 E8 First on dev_a1: 22.7->22.3, -0.39+-1.61 (8/9), cut 27.0->28.5. First won only 42/256, Let->Here 10 (counter-write overshoots for First). First group n42: E8 19.0, orig 26.2, E5 23.8 (selected, weak evidence). Register not fairly tested; needs about +2-3 logits, near RMS ceiling, second site if pursued.
EDIT E9 Okay-opener (pass 24): L18 down_proj column 2543 (from ORIGINAL) += d/abar, d=1.7*Sb*(unit(gxp)-kap*unit(glh)), row Okay, Sb=1.3221, kap=0.089 (kap_rel .10). Generate both on /work/dev_a1.json. Bar: beat E5 23.9 by about 3. Copy /work/cands/e9_okay. E5 stays in best_1. Queue: The.

Pass 24 E9 Okay on dev_a1: 22.7->22.3, -0.39+-2.38 (18/19), cut 27.0->19.9, 198 Okay, style 'Okay, let's break down the problem step-by-step: 1. **...' (list family). Okay group: E9 19.2, orig 20.7, E5 22.2. Natural Let 30.9->16.0 (harmful). Natural Here 13.7->21.4 (E5 ~19.7). Flipped by E5: Okay 18.5 vs Let 19.8 (11/12). m<-2 (n29, POST HOC): orig 24.1, E5 27.6, Okay 37.9.
Pass 25 plan: E9 unchanged, active. Generate both on /work/dev_k.json (149 problems kept in Here by E5 from dev_s2,dev_a2,dev_s3 plus dev_s1 m<-1.9, dev_a1 excluded). Prediction: Okay gains >=5 pts on these. If near zero or negative, Okay closed. Three-way policy idea (strongly Here to Okay, rest as E5) depends on this. Queue: The.
dev_k filtered: removed 4 problems with known margin above -1.5, now 145 problems. E9 still active, pass 25 generate both on /work/dev_k.json.
dev_k final: 136 problems, only Here's under E5 (dev_s2,dev_a2,dev_s3) or dev_s1 m<-1.9. E9 active, pass 25 generate both on /work/dev_k.json.

Pass 25 E9 on dev_k (136 fresh strongly-Here): 24.3->22.1, -2.21+-2.44 (4/7), cut 9.6->12.5. Prediction failed, 37.9 was noise. OKAY AND THREE-WAY POLICY CLOSED. E9 dropped, no active edits. E5 in best_1 still best. All registers tried: Let >= We > Okay ~ Here overall.
New line: premature finish (28% of wrong finishes have fewer eqs than ref, 10% end on an intermediate). First check selectivity of finish confidence on pass 2 topk (free).

Free analysis (run 81-82, all 1250 original answers): Let answers end 'Final Answer: ...boxed' with marker only .4%, Here use #### 75%. Extraction loss ~0: gold in last line of wrong finished <=.3%. EXTRACTION LINE CLOSED. Premature finish is a Here-mode trait: short among wrong Here .36, Let .04; median eqs Here 3, Let 8. PREMATURE-FINISH LINE CLOSED for Let, E5 already removes most. Finish confidence no selectivity (n small). Next: loop BREAKING (not stopping) in Let mode.
Loop characterisation: Let loops 93% exact repeat, onset ~25% through answer, period 1-8, restate earlier lines after dead end. In-loop vs pre-loop neurons: 1145-1613 of 6912 |t|>6, smooth ranking, confounded with confident copying, no isolated component. LOOP-BREAK LINE CLOSED without a pass.
Run 87 proposal refused by review (no reason given): a screen of later-layer neurons comparing prompt activity with mean answer activity. Do not retry. Use L21 n5695 from the pass 1 note; learn side effects only via forward passes.
EDIT E10 To-opener two sites (pass 26): L18 down_proj col 2543 (from ORIGINAL) += d18/abar, d18=1.7*Sb*(unit(gxp)-kap*unit(glh)), row To, Sb=1.3255, kap=0.089; L21 down_proj col 5695 (from ORIGINAL) += d21/ab2[5695], d21=1.0*Sb2*unit(gxp2), gain2=(1+w21)(1+wf), Sb2=1.3546, ab2 = mean last-prompt activation pass 1. TF /work/tf13_texts.json topk 5, compare pass 13. Copy /work/cands/e10_to. E5 stays in best_1.

Pass 26 E10 TF: To top 230/1024, Let 657, Here 137, gap -1.49 (q05 -4.38, q95 .75), shift >=10.85, Let-Here drift +1.26. Leak: only token 3 back +1.12.
EDIT E10b (pass 27): as E10 with L18 mult 2.2 kap 0.089, L21 mult 2.5 with counter term: d21=2.5*Sb2*(unit(gxp2)-kap2*unit(glh2)), kap2=0.071. Generate both on /work/dev_a1.json. Hypothesis: goal-first register beats E5 23.9 by about 3. Copy /work/cands/e10b_to. E5 in best_1.

Pass 27 E10b To two sites on dev_a1: 22.7->18.0, -4.69+-2.70 (18/30), cut 27.0->25.0. To 167/256 (160 'To find'), Let 89. To group: E10b 17.4, orig 20.4, E5 22.8. Kept-Let group n89: E10b 19.1, orig 27.0, E5 25.8 (opener unchanged yet much worse, selected group; possible answer-token damage from strong writes, learn only via passes). Natural Let 30.2->18.0. TO / GOAL-FIRST REGISTER CLOSED, first-line overlap was correlation. E10b dropped.

Free content stratification (runs 93-95, 1250 orig answers, control ref steps, 10 features, selected): decimal -9.8+-2.8, large>=1000 -8.2+-3.1, time -6.1+-1.6, rate -4.5+-1.4, time&rate -8.5+-1.8 (time only -2.4, rate only -.7). True hidden operand (not in text or earlier result, word numbers mapped): n391 acc 12.3 obs-exp -7.9+-1.6; time constants n114 acc 8.8. Answers use constant ~50%, acc|uses only 15-19. Dropped factor: T&R short 6.5 = over 6.5, other 82: DROPPED-FACTOR LINE CLOSED. Difficulty description, no lever yet. Global (1+w) q_norm rescale ruled out as whole-tensor scale in disguise; only head-specific q_proj rows if sharpening is pursued.

Run 98 proposal refused by review (no reason given): per-token vote of attention heads vs model top-1, framed as predicting whether sharpening could help. Do not retry in any form. Learn sharpening effect only by forward pass.

EDIT E11 head sharpen (pass 28): L17 q_proj rows d in head 0 (row index d<256) zeroed for 44 dims with |(1+wq)(1+wk)|<1 (list /work/cands/e11_dead_dims.json). Weights only. Row energy share 0.386, logit energy share 0.0011, estimated head 0 temperature x1.277. Only active edit. TF /work/tf7_texts.json, logprobs, topk 5, compare pass 7. Prediction: fromproblem_first +0.5 or more, dlp>0. Copy /work/cands/e11_h0sharp. E5 stays in best_1. HOLDOUT unused; keep at least 4 passes for it.

Pass 28 E11 L17 h0 sharpen x1.28 TF tf7 vs pass 7: fromproblem_first -0.18+-0.56 dlp -.163+-.015; fromearlier_first -1.50+-.49 dlp -.078; word -.35+-.10; result_later +.51+-.20; +3 +1.67+-1.66; all arith +.15+-.17; mean text lp -.0195+-.0007. PREDICTION REFUTED: sharpness moves confidence, not choice. Weight fact: 44 dead attention dims at L17 hold 38.6% of h0 row energy, .11% logit energy, a usable temperature knob. Flatter dose not run yet. E11 dropped.

FROZEN E5 for HOLDOUT: L18 down_proj col 2543 = original + (0.6/1.5)*E1 delta, file /work/best_1. Passes 29-32: generate both on /work/hold_1..4.json (256,256,256,232, all 1000 HOLDOUT). Read pooled paired gain only, no tuning from it. DEV evidence before seeing HOLDOUT: pooled +1.7+-1.0 on 994. Then 18 passes left for exploration.

HOLDOUT E5 passes 29-32 (1000): orig 20.90, E5 20.50, gain -0.40+-0.98, 46/50. E5 NOT CONFIRMED, does not beat original on HOLDOUT. Edits dropped, /work/edited empty: original is current submission unless a new candidate beats it on HOLDOUT. Do not tune on HOLDOUT, flipped/kept split read for confirmation only.

E5 HOLDOUT confirmation read: flipped n354 -0.56+-2.30 (DEV +5.81+-2.85), kept n646 -0.31+-.85, flipped cut 22.9->30.8; all data +0.65+-.70 is context only. MODE/REGISTER LINE CLOSED FOR GOOD. Lesson: 1-2 pt DEV gains not trustworthy; want well above that.
EDIT E11b (pass 33): L17 q_proj head 0 rows of the 44 dead dims x1.567 (from ORIGINAL), est head 0 temp x0.8. Only active edit. TF /work/tf7_texts.json, logprobs, topk 5, compare pass 7. Continue only if word and earlier-recall rise by >=2 SE with no class falling. Copy /work/cands/e11b_h0flat. Original is current submission. Keep 4 passes for HOLDOUT of any new candidate.

Pass 33 E11b L17 h0 flatter x0.8 TF tf7 vs pass 7: fromproblem_first -1.90+-.42, fromproblem_later -.76+-.22, word -.23+-.07, arith all -.46+-.21, result_later -.56+-.20, text lp +.0019. Fails rule. Two-sided dose: flatter hurts retrieval, sharper adds nothing, natural at plateau. ATTENTION SHARPNESS L17 h0 CLOSED, causal role of h0 shown. E11b dropped, no active edits, original is current submission.

Pass 34 plan: teacher_forced ORIGINAL, no edits, /work/tf34_texts.json (meta tf34_meta.json): 2 texts, each 2 own cut Let answers with exact token periodicity (mechanical pick, 4 loops), logprobs, topk 3, attn_layers 5,11,17,23. Purpose: does one head lock onto the copy position one period back. Lock-on head would get a removal test in generation (DEV). Spread means no copy circuit, close loops for good. Original is current submission.

Pass 34 attention in 4 exact loops (P 16,45,79,143): copy lock-on heads, attention on next token in earlier repeats: L5h2 .452, L17h3 .401, L17h2 .390, L17h0 .386, L5h3 .314, L11h1 .287; off-target ~.003; prelag .03-.07; L23 none. Overlaps retrieval/operand heads: looping is the normal pointer-copy circuit aimed at own text, not a separate circuit. L5h3 has no known retrieval role, first candidate.

Pass 35 plan: teacher_forced ORIGINAL, no edits, /work/tf35_texts.json (meta tf35_meta.json), 205 periodic own loops + 250 correct + 569 wrong finished DEV answers, logprobs, topk 3. Baseline for loop-escape screen: escape = top1 departs from copied token within first repeat; damage = top1 departs from own correct answer tokens. Candidates next: L5 h3 x0, L5 h2+h3 x0, L11 h1 x0. Original is current submission, keep 4 passes for HOLDOUT.

EDIT E12 (pass 36): L5 o_proj columns 512..1023 (heads 2,3) zeroed, from ORIGINAL, only active edit. TF /work/tf35_texts.json, logprobs, topk 3, compare pass 35 with screen(). Good outcome: loop escape >10 pts above baseline and clearly above divergence on correct answers. Copy /work/cands/e12_l5h23. Original is current submission.

Pass 36 E12 L5 h2+h3 x0 TF tf35: per-token mismatch first repeat 1.71, late loop .40, correct 3.01, wrong 4.21, pre-loop 4.25 (baseline .07,.00,.27,.37,.33). Any-mismatch escape 65.9 is saturation, NOT selective (metric lesson: same units). Copying redundant, loops robust. LOOP-BREAK BY HEAD REMOVAL CLOSED. E12 dropped, copy in cands.
Next idea: prompt-reading heads during answers (L11h3,L5h1,L11h2,L23 in loops), never boosted. First free: what do they read in pass 34 pre-loop tokens.

Prompt attention split (pass 34 pre-loop): no dedicated problem reader, body attention spread .1-.25 over many heads, L11h3 and L17h1 mostly specials. Gain boost of prompt heads dropped (plateau seen in E2,E3,E11,E11b).
New hypothesis: balance of reading problem vs own text. L17 h0 ~.41 own text, ~.27 problem. Pass 37 plan: teacher_forced ORIGINAL, no edits, /work/tf37_texts.json (meta tf37_meta.json, 128 correct + 128 wrong own DEV answers), hidden_states 4,10,16, no topk. Purpose: regional mean inputs to build a prompt-vs-answer key direction for a constant query bias (rank-1 on q_proj, head 0). Then TF tf7 vs pass 7: fromproblem_first should rise, fromearlier_first may fall. Original is current submission, keep 4 passes for HOLDOUT.

EDIT E13 (pass 38): L17 q_proj head 0 += outer(c,n), from ORIGINAL, only active edit. u=KP-KA (mean key problem minus answer incl k_norm gains, pass 37 regional means), c=s*gq*u with target separation D=1.0 logit (S=16, rq=.598 estimated), n=d/(d.mA), d=mA minus projection on mP so n.mA=1, n.mP=0. Vectors in /work/cands/e13_vectors.npz. TF /work/tf7_texts.json, logprobs, topk 5, compare pass 7. Prediction: fromproblem_first +0.5 or more, fromearlier_first may fall. Bug caught before saving: c used gq^2, fixed to gq. Copy /work/cands/e13_h0bias. Original is current submission.


Pass 38 E13 L17 h0 problem bias (D=1) TF tf7 vs pass 7: fromproblem_first +.84+-.27 dlp +.028, othernum_first +1.19+-.51, fromearlier_first -1.65+-.31 dlp -.035, fromearlier_later -.18+-.13, result_later +.31+-.14, word -.02+-.04, arith all +.05+-.12, text lp -.0003. Prediction MET with the expected cost. First attention edit that moves source choice, not only confidence. Count-weighted net over number classes +4.9+-11.4 of 12588 tokens: a trade, near zero under teacher forcing.
Plan: generate both E13 on the least-used DEV set (exact set in next note). Support needs about +4 or more, given the E5 lesson. Near zero closes the uniform bias. Original is current submission, keep 4 passes for HOLDOUT.

Pass 39 plan: generate both, E13 only active edit (L17 q_proj head 0 += outer(c,n), D=1, /work/cands/e13_h0bias, md5 57c2a7ae), problems /work/dev_a2.json (256, least-used full DEV set, mean 1.12 earlier uses). Rule: about +4 or more, pool with dev_s3; near zero or negative closes the uniform bias. Original is current submission, keep 4 passes for HOLDOUT.

Pass 39 E13 generate both dev_a2: orig 13.67, E13 12.89, -0.78+-1.11, flips 3/5, cut 28.5->27.7. Diverged 147/256; needed-number coverage on diverged 0.9202->0.9360 (+0.0158+-0.0082). Below the +4 rule: UNIFORM READING BIAS CLOSED as a candidate. Natural problem/own-text balance appears near its optimum (TF net +4.9+-11.4). E13 dropped, copy in cands, original is current submission.

Deeper weight scan (runs 124-125, weights only): kurtosis, effective rank, top singular share, min unit norm, near-null eigen, norm gains <=0, single-dim jumps. All flags follow layer type or depth: full-attention layers share heavy-tailed k_proj (18.1,8.8,8.2,5.7), low-rank q_proj (erank 212-248 vs 270-430) and a few near-null q directions; L17 small q rows = known low-gain dims; early layers heavy-tailed MLP outputs, negative norm gains spread. NO FAULT FOUND: capacity, not damage.
Multi-term sums in 1250 original DEV answers (runs 126-127, clean boundaries): 3-term 226 eqs 24.3% wrong, 4-term 59 78%, 5+ 29 96.6% (mostly loop repeats); drop-last 10.9% of wrong; answers with a wrong 3+-term sum 5.5% (acc 4.3); true sum = gold in 13 (1.0% of answers). Perfect fix ceiling ~1-2 pts: MULTI-TERM SUM LINE CLOSED as a lever (garbled, not dropped).

Self-correction phrases (run 128): Wait 4/1250, mistake/try again 0, re-examine 26 (96% cut, part of loops). NO RECOVERY BEHAVIOUR TO STEER, closed.
Pass 40 plan: teacher_forced ORIGINAL, no edits, /work/tf40_texts.json (949 tf7 texts with <=443 tokens: 450 refs, 499 arith; tf7 index map in tf40_meta.json), hidden_states 17,20,22,24,26, logprobs, topk 5. Purpose: depth diagnostic (logit lens = projection of hidden state times final-norm gain on candidate token rows; argmax unaffected by RMS). Layer 26 must reproduce the tool's top-1 (calibration). Question: at failing positions, does the reference token lead at L22/L24 and lose late (late overwrite, repairable) more often than correct positions show the reverse? If never leads: error formed, capacity. Original is current submission.

Pass 40 depth diagnostic (original, 949 tf7 texts; lens = hidden x final-norm gain on digit rows; calibration h26 vs tool first digit 99.57%). INDEX NOTE: hidden_l = output of model.layers.(l-1) (hidden_26 = after layers.25), so hidden_17 is the INPUT of full layer 17; E13 used hidden_16 (input of layer 16) as layer-17 input, a one-layer approximation (TF effect still as predicted). Digit acc by depth (h17,h20,h22,h24,h26): fromproblem_first 27.3,78.4,78.5,79.0,79.0; fromearlier_first 33.9,86.8,87.3,87.1,87.4; result_first 26.2,78.3,90.7,92.9,93.3; result_later 46.7,79.5,90.4,93.6,94.4. Late flips balanced for operands (fromproblem_first R24W26 41 vs W24R26 41), net positive for results (all R22W26 101 vs W22R26 242). Operand choice FORMED in layers 17-19 and never repaired or corrupted later; results refined to the end. LATE-OVERWRITE LINE CLOSED ('right at any earlier layer' 33% is chance at h17).
Next (free): pass 5 attention of L17 h0 at wrong operand picks: peak on the source of the predicted (wrong) number = upstream plan error; peak on correct source = downstream mistranslation (possible lever).

Pass 5 attention re-read (run 133): L17 h0 at fromproblem first digits, 71 positions. Wrong picks n17: top-read number starts with the PREDICTED digit 76%, correct source 24%; mass predicted .302 vs correct .127. Right n54: correct source 76%. The head faithfully retrieves what the query asks for: wrong operands are an UPSTREAM PLAN error (query formed before L17), downstream mistranslation ~1/4. Explains why gain/sharpness/balance edits of h0 only trade.
Pass 41 plan: teacher_forced ORIGINAL, no edits, /work/tf41_texts.json (2 packed texts of fresh DEV problems + reference solutions, ids in tf41_meta.json), logprobs, topk 5, attn_layers 5,11,17,23. Purpose: replicate the 3:1 split with about double the sample. Original is current submission.

Run 135 proposal REFUSED by review (no reason given): all 16 full-layer heads, top-read number vs model's predicted digit at wrong picks, framed as finding a dissenting head to boost. Same family as run 98 (per-token head vote vs model prediction to predict an edit). Do not retry in any form; DISSENTING-HEAD LINE CLOSED without data. Only the cleared run-133 description of L17 h0 is replicated on pass 41.

Pass 41 replication (L17 h0, fresh DEV refs) pooled with pass 5: P wrong n29 predicted 62%, correct 28%, other 10% (mass .250 vs .125); E wrong n55 predicted 69%, correct 18%, other 13% (mass .280 vs .115); right picks correct 81% (P n94, E n251). Pass 41 alone P wrong n12 weaker (42/33/25). CONFIRMED: about 2/3 of wrong operand picks are faithful retrievals of the wrong intended number (plan error upstream of L17), about 1/5-1/4 read the correct source but output another digit. Account: failures start with the wrong intention; retrieval executes it; late layers keep it.

EDIT E13x2 (pass 42): L17 q_proj = original + 2*(E13 - original), rank-1 on head 0 rows only, target separation D=2. Copy /work/cands/e13x2. Generate both on /work/dev_s3.json (226). Mechanism test: prediction coverage of needed numbers among diverged up ~3 pts, accuracy flat or down. Accuracy >= +4 would contradict the plan-error account and earn HOLDOUT. Original is current submission.

Pass 42 E13x2 generate both dev_s3 (226): orig 20.80, E13x2 21.24, +0.44+-1.93, flips 10/9, cut 23.5->25.7. Coverage on diverged d +0.0071+-0.0066 (D=1 pass 39: +0.0158). Not a candidate (far below +4). E13x2 dropped, copy in cands; no active edits, original is current submission.

E13 mechanism test read: D=2 coverage on diverged +0.71+-0.66 pts (D=1 dev_a2 +1.58+-0.82); pooled +1.12+-0.52 pts on 302 diverged; accuracy pooled about -0.4+-1.0. Prediction (+3 pts coverage) NOT MET: the bias moves number use only slightly in free generation. Consistent with the plan-error account but a weak test of it; the account rests on the attention (passes 5,41) and depth (pass 40) results.
Pass 43 plan: teacher_forced ORIGINAL, no edits, /work/tf40_texts.json, hidden_states 18,19,21,23,25 (outputs of layers 17,18,20,22,24), logprobs, topk 5. Purpose: complete the depth profile (with pass 40: every layer output 16..25). Question: is operand choice made inside layer 17 (h18 ~78% for fromproblem_first) or built over 18-19? Original is current submission. Then: E4 (Let->Here, frozen, cands/e4) on HOLDOUT as a held-out confirmation of the register's causal effect (not a candidate).

Run 139 proposal REFUSED by review (no reason given): logit lens of pass 43 hidden states (final-norm gain and tied output rows applied to stored states). Likely judged as running the output head in the kernel. Do not retry in any form; LENS LINE CLOSED. The pass 40 depth result (run 131, cleared) is marked POSSIBLY CONTESTED; it was descriptive only and chose no edit. Account rests on pass evidence: L17 h0 attention (passes 5, 41) and paired generation results.
Plan passes 44-45: E4 (frozen, cands/e4: L18 down_proj col 2543 = original - (2.0/1.5)*E1 delta, Let->Here) generate both on /work/hold_1.json and /work/hold_2.json. HOLDOUT confirmation of the register's causal effect (DEV -6.25+-2.18); not a candidate, nothing chosen on HOLDOUT. Original is current submission.

Pass 44 E4 (Let->Here, frozen) HOLDOUT hold_1 (confirmation only): 20.31 -> 17.97, -2.34+-2.07, flips 11/17, cut 25.0->22.3. Split: flipped Let->Here n109 20.2->11.9 (-8.26+-4.15), Let kept n28 35.7->35.7, natural Here n119 16.8->19.3 (+2.52+-1.87; DEV not-flipped -1.3, read as noise). REGISTER CAUSAL EFFECT REPLICATES on held-out problems (DEV flipped 24.3->10.7). Reverse flips do not help (E1 DEV, E5 HOLDOUT flipped -0.56+-2.30): the model's own register choice is about optimal.
Pass 45 plan: E4 unchanged (only active edit), generate both on /work/hold_2.json, confirmation only (pool flipped group with pass 44). Original is current submission; E4 is not a candidate. Remaining after: 5 passes; next idea: HOLDOUT attention replication of the L17 h0 2/3 split (descriptive, cleared analysis of run 135).

Pass 45 E4 HOLDOUT hold_2: 20.70 -> 19.53, -1.17+-1.88 (10/13). Split: flipped n104 22.1->18.3 (-3.85+-4.30), Let kept n34 -2.94+-2.94, natural Here n118 +1.69+-1.19. POOLED HOLDOUT hold_1+2 (512): flipped Let->Here n213 21.1->15.0, -6.10+-2.98 (DEV -13.6 on n103): REGISTER EFFECT CONFIRMED ~2 SE on held-out problems. Overall E4 on 512 about -1.8.
Unplanned HOLDOUT observation: natural-Here problems under E4 n237 17.3->19.4, +2.11+-1.11. Derived from HOLDOUT: NOT used for choosing. From now on hold_1/hold_2 count as spent on exploration; hold_3/hold_4 are the only untouched held-out sets for any confirmation. Check DEV pass 10 split (free) before anything else.

DEV pass 10 E4 split (run 143): flipped n103 24.3->10.7 (-13.59+-4.79), Let kept n45 0.00, natural Here n108 21.3->19.4 (-1.85+-1.85). HOLDOUT natural-Here +2.11+-1.11 not reproduced on DEV; combined about +1.1+-0.95: NOISE, closed.
Pass 46-47 plan: E4 unchanged (only active edit, frozen), generate both on /work/hold_3.json then /work/hold_4.json. Purpose: complete held-out measurement of the register effect on flipped problems (target SE ~2). No candidate exists, so hold_3/hold_4 are not needed for a candidate. Then pass 48: attention replication on HOLDOUT refs (descriptive, run 135 analysis). Original is current submission.

Pass 46 E4 HOLDOUT hold_3: 19.92 -> 18.75, -1.17+-2.18 (14/17). Split: flipped n93 -5.38+-4.92, Let kept n34 0.00, natural Here n129 +1.55+-2.20. POOLED HOLDOUT hold_1-3 (768): flipped n306 21.9->16.0, -5.88+-2.56 (2.3 SE); Let kept n96 -1.04+-2.34; natural Here n366 +1.91+-1.05 (DEV -1.85+-1.85; all data ~+1.0+-0.9, unconfirmed observation, no held-out data left to test a derived candidate, not pursued).
Final pass plan: 47 E4 on /work/hold_4.json (complete n=1000); 48 drop edits, attention replication on HOLDOUT refs (2 packed texts, attn 17); 49-50 teacher-forced HOLDOUT reference solutions, original then E11b (L17 h0 flatter x0.8, cands/e11b_h0flat), causal confirmation of h0 retrieval role (DEV: fromproblem_first -1.90+-.42). Original is current submission.

Restart check (run 150): /work/edited == cands/e4 (L18 down_proj, only col 2543 differs from original). Pass 47: E4 frozen, generate both on /work/hold_4.json (232 HOLDOUT), confirmation of the register effect only, nothing chosen. Original is current submission; drop all edits before finishing.

Pass 47 E4 HOLDOUT hold_4 (232, confirmation only): 22.84 -> 19.83, -3.02+-2.06 (8/15), cut 24.1->20.7. Split: flipped n87 20.7->16.1 -4.60+-4.88; keptLet n37 35.1->35.1 +0.00+-0.00; natHere n108 20.4->17.6 -2.78+-2.06. POOLED HOLDOUT hold_1-4 (1000): flipped n393 21.6->16.0 -5.60+-2.26; keptLet n133 30.8->30.1 -0.75+-1.69; natHere n474 17.5->18.4 +0.84+-0.94; all n1000 20.9->19.0 -1.90+-1.02.
E4 dropped after pass 47 (copy in cands/e4); no active edits; original is current submission. Next: pass 48 attention replication on HOLDOUT refs (original), passes 49-50 TF HOLDOUT refs original vs E11b.

Pass 48 plan: teacher_forced ORIGINAL, no edits (edited() empty), /work/tf48_texts.json (2 packed texts of HOLDOUT problems + reference solutions, same recipe as tf41, seed 20261110, ids in tf48_meta.json), logprobs, topk 5, attn_layers [17]. Purpose: held-out replication of the cleared L17 h0 description (collect_h0, block 135): share of wrong operand picks where h0 reads the predicted (wrong) number vs the correct source. Descriptive only, nothing chosen. Original is current submission. (Run 151 overwrote helper E with a dict; restored from block 11 in plain code, run 156.)

Pass 48 result (HOLDOUT L17 h0, collect_h0 unchanged): P HOLD wrong n8: correct 0% predicted 62% (se 17) other 38% | mass ok 0.068 pred 0.164 || P HOLD right n53: correct 92% predicted 2% other 6% | mass ok 0.475 pred 0.019 || P ALL wrong n37: correct 22% predicted 62% (se 8) other 16% | mass ok 0.112 pred 0.232 || P ALL right n147: correct 85% predicted 5% other 10% | mass ok 0.431 pred 0.033 || E HOLD wrong n19: correct 26% predicted 58% (se 11) other 16% | mass ok 0.106 pred 0.257 || E HOLD right n107: correct 90% predicted 3% other 7% | mass ok 0.396 pred 0.041 || E ALL wrong n74: correct 20% predicted 66% (se 5) other 14% | mass ok 0.112 pred 0.274 || E ALL right n358: correct 84% predicted 4% other 12% | mass ok 0.397 pred 0.040 || PE HOLD wrong n27: correct 19% predicted 59% (se 9) other 22% | mass ok 0.095 pred 0.229 || PE HOLD right n160: correct 91% predicted 2% other 7% | mass ok 0.422 pred 0.033 || PE ALL wrong n111: correct 21% predicted 65% (se 5) other 14% | mass ok 0.112 pred 0.260 || PE ALL right n505: correct 84% predicted 5% other 11% | mass ok 0.407 pred 0.038

Pass 48 read: HOLDOUT wrong picks n27 h0 reads predicted 59% (se 9), correct 19%; right picks correct 91% (n160). All data wrong n111 predicted 65% (se 5), correct 21%. PLAN-ERROR ACCOUNT REPLICATES ON HOLDOUT.
Pass 49 plan: teacher_forced ORIGINAL, no edits, /work/tf49_texts.json (all 1000 HOLDOUT reference solutions in the eval prompt, built as tf7 refs in block 26; meta tf49_meta.json), token_logprobs, topk 5. Baseline for pass 50 (E11b, L17 q_proj head 0 dead dims x1.567, cands/e11b_h0flat) = held-out causal test of h0 retrieval (DEV pass 33: fromproblem_first -1.90+-.42). Analysis: CL=classify(meta49, fwd_49), class table as compare7 first half. Original is current submission.

Pass 50 plan: EDIT E11b only (L17 q_proj head 0 rows of the 44 dead dims x1.567 from ORIGINAL, file cands/e11b_h0flat, est head 0 temp x0.8). teacher_forced EDITED on /work/tf49_texts.json, token_logprobs, topk 5; compare with pass 49 (original, same texts) by class (classify, compare7 first half). Held-out causal confirmation only, nothing chosen. AFTER PASS 50: drop E11b so /work/edited is empty; submission = ORIGINAL.

Pass 50 E11b vs pass 49 (HOLDOUT refs, 1000): fromearlier_first n4534 0.8796->0.8778 -0.18+-0.22 dlp -0.0056+-0.0025 || fromearlier_later n4994 0.9876->0.9868 -0.08+-0.10 dlp -0.0241+-0.0018 || fromproblem_first n4878 0.7839->0.7673 -1.66+-0.28 dlp -0.0252+-0.0038 || fromproblem_later n3169 0.9675->0.9631 -0.44+-0.18 dlp -0.0371+-0.0037 || othernum_first n1511 0.6519->0.6479 -0.40+-0.42 dlp +0.0587+-0.0091 || othernum_later n932 0.8251->0.8240 -0.11+-0.28 dlp +0.0469+-0.0117 || result_first n3587 0.9264->0.9231 -0.33+-0.15 dlp -0.0078+-0.0012 || result_later n4140 0.9495->0.9473 -0.22+-0.12 dlp -0.0120+-0.0014 || word n66513 0.6832->0.6811 -0.21+-0.05 dlp -0.0019+-0.0007 || text lp +0.0039+-0.0004. DEV pass 33: fromproblem_first -1.90+-.42.
FINAL: E11b dropped, /work/edited empty, SUBMISSION = UNEDITED ORIGINAL. Best candidate E5 (L18 down_proj col 2543, alpha +0.6): DEV +1.7+-1.0 (994), HOLDOUT -0.40+-0.98 (1000), not confirmed.
