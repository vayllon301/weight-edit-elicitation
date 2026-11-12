# NOTES
## Sets
DEV = /work/dev_ids.json (1250 ids, seeded split rng 12345, first 1250 of permutation)
HOLDOUT = /work/holdout_ids.json (1000 ids) -- only for confirmation, never tune on it.
## Current edit
none
## Log
Pass 1: generate, model=original, problems=/work/dev512a.json (DEV[:512]), activations layers 0..26 + mlp. Current edit: none.
Pass 1 result: original DEV[:512] acc 22.3%, 277 wrong, 121 cut (loops). Failures look like a weak solver (misreading, dropped steps), arithmetic ok.
Pass 2: teacher_forced original, /work/tf1_texts.json (500 prompts: G1 math+instr, G2 math only, G4 math+translate, G3 nonmath, G5 nonmath+instr; meta /work/tf1_meta.json), hidden all layers. Current edit: none.
FINDING: layers 10,11,12 up_proj each have ~30-45 rows with a huge component (|proj| 0.8-1.4, bulk 99% <0.12) on a shared direction v0 (top right-sing. vec of up_proj; cos across layers 0.82-0.89), both signs. Looks planted.
Pass 3: both, DEV[:256] (/work/dev256a.json). Edit E2: layers 10,11,12 up_proj rows with |W@Vt0|>0.3 -> remove projection on Vt0 (frac 1.0). rows in /work/E2_rows.json
Pass 3 result: E2 (remove v0 comp. from L10-12 up rows) = -3.1 +- 1.65 (5 w->r, 13 r->w). v0 group writes are ANTI-coherent (decoy-like). Dropped E2.
Pass 4: teacher_forced original, /work/tf2_texts.json: reference solutions (DEV[256:512]) under prompt variants V1 eval, V2 problem only, V3 +'Solve step by step.', V4 +'The answer is <number>' format; meta tf2_meta.json (variant,id,prompt_len,prompt+body_len). Current edit: none.
Weight forensics closed (runs 37-47): norms natural, no zero singular values, no swapped heads (QK/OV cross-layer match), v0 group = constant (bias-like) read, not math-gated, writes random -> decoy. No unit reads math dir (max|cos| .21).
tf3 screening set: /work/tf3_texts.json = DEV[:1024] reference solutions under eval prompt; meta tf3_meta.json (id,prompt_len,prompt+body_len).
Pass 5 (planned): both, DEV[:256] (/work/dev256a.json). Edit D1: math-vs-nonmath direction dm[10] (last prompt token, normalized, massive dims removed; /work/D1_dir.npy) projected out of ALL 26 layers' o_proj and down_proj outputs, folded through post-norm scale (u=r*(1+w_post)). Strength full (1.0).
Pass 5 result: D1 (math dir dm[10] ablated from all o/down writes) = +0.8 +- 2.8 (26 w->r, 24 r->w). Cuts 58->21, marker 26%->85%: D1 switches answer STYLE (algebraic LaTeX 'Let $x$...' -> markdown) but no accuracy change; both styles ~20-25%. Math-mode/style is not the brake. D1 kept off.
Pass-1 MLP answer means: many layers have one near-constant dominant writer (L23 n5826 46%, L16 n3437 40%, L10 n1868 36%, L8 n830 34%, L7 n5434 33%).
Pass 6 (planned): teacher_forced original, /work/tf6_texts.json (30 each G1 math+eval instr, G4 math+translate, G3 nonmath, G5 nonmath+instr; meta tf6_meta.json), mlp_act all 26 layers -> math selectivity of neurons. Current edit: D1 still saved but irrelevant (TF on original).
Pass 6 result: dominant answer-time MLP writers are generic (same on math/nonmath). 66 'solve-math' neurons (G1-only, L6-23) fire only at last prompt token, ~silent in answers -> opening/style units, not a brake. MLP unit-level brake line closed.
Pass 7 (planned): teacher_forced original, /work/tf7_texts.json (A: gsm8k_train_5465 eval prompt + model's own correct md answer; B: nonmath ocean-poem prompt + eval instr + poem), attn_layers [5,11,17,23] + token_logprobs. Current edit: D1 still saved (irrelevant for original pass).
Pass 7 result: global-layer attention (5,11,17,23) healthy: number-retrieval heads L17h0 .66, L5h2 .47, L17h2 .36; instruction attention <=.07 all heads. Attention-brake (global) line closed.
Runs 59-61: digits by role from pass 4: copied_cont -0.13, result_first -0.34, copied_first/othernew_first -0.73 -> copy & arithmetic ok, deficit in choosing quantities. Wrong answers contain gold only 13-16% (no final-step sabotage, no format artefact).
Pass 8 (planned): teacher_forced original, /work/tf8_texts.json (= tf3[:150], DEV[:150] reference solutions, eval prompt), hidden_states 12..26 + topk 20 + token_logprobs -> restricted logit lens (overthinking/late-brake test). Current edit: D1 saved but irrelevant.

Pass 8 result (runs 63,75): restricted logit lens L12-26 on DEV[:150] ref solutions: logprob of ref tokens rises ~monotonically to final layer (result digits top1 .29@L18 -> .91@L26; copied first digit jump at L18). No late brake/overthinking. Line closed.

Run 78: arithmetic slips (chains excluded): orig 7.1% of explicit calcs wrong, 6.2% of answers; D1 4.2%/4.7%. Real but small -> not the main deficit.
Run 79: D1 archived to /work/cand_D1 and DROPPED. Current edit: none.

Run 83: pass-7 attention: L11h2/h3 put ~26% of answer-token attention on the LAST prompt token (sink-like; BOS only .04/.10) -> broadcast channel of end-of-prompt state.
Run 85: the 66 G1-only 'solve-math' neurons (/work/U1_neurons.json, L6-23) write ~orthogonal to D1 (|cos|<=.18), large share of G1-G5 diff in L11 (.70), L12 (.61), L17/19/20 (~.3).
Pass 9 (planned): both, DEV[:256] (/work/dev256a.json). Edit U1: down_proj columns of the 66 neurons in /work/U1_neurons.json zeroed (17 tensors model.layers.{6..23}.mlp.down_proj), strength full.

Pass 9 result: U1 (66 end-of-prompt solve-math neurons zeroed) = -0.4 +- 2.25 (16 w->r, 17 r->w); cuts 58->37, shorter answers. Style/length units, not a brake. CLOSED. Edit U1 still saved (to drop).
Runs 87-109 (no passes), all CLOSED:
- v0 group (L10-12 up rows): not gate/up swap; = natural row + added v0 comp (norm arithmetic fits) BUT same pattern in L18-19 up, L23-24 gate, L25 up (83 rows) -> natural Gemma feature. No shared gate/down direction, no engineered pairs. CLOSED.
- math-content-selective neurons (prompt tokens, all layers): diffuse, both signs, natural. Correct-vs-wrong neuron d: smooth tails, no outlier brake.
- error types: 78% generic 'other'; 10-12% intermediate; operator choice under TF 70-89% correct, errors spread. Omitted needed numbers ~10%, uniform across sentence positions.
- accuracy by #steps: 1-2 steps only 29-38% (!) -> broad weakness (binding/relations), even easy problems.
- per-layer update norms math/nonmath ratio 0.91-1.08 all layers; no RMU-like fixed-vector push (pass 8 L12-26 solution vs prompt similar).
- math contrast jumps at L7 (attention-driven, MLP write diffuse, cos~0).
- RoPE band capacities smooth; norm scales smooth with depth; per-head o_proj norms 0.69-1.63x (no damping).
- heads with dominant OV dir = sink writes along massive/mean dirs (L11h2/h3, L13, L17h1, L18, L21h2, L14h1); L22h0 exception = formatting head (+'**', -EOS/-eot/-newline).
PLAN: matched-pair contrast (same problem, correct vs wrong answers across variants E2/D1/U1/original) via TF on original -> difficulty-controlled correctness direction; look for a component writing it. Default if nothing: submit ORIGINAL.
Pass 10 (planned): teacher_forced ORIGINAL, /work/tf10_texts.json (67 matched pairs: same DEV problem, correct vs finished-wrong answer from variants orig/E2/D1/U1; meta tf10_meta.json [id,label,tag,prompt_len,total_len,same_style]), hidden_states [4,8,10,12,14,16,18,20,22,24,26] + token_logprobs. Saved edit U1 irrelevant (original model).

Run 116: dominant near-constant answer-time writers: L7/L8/L10 98% massive-dim (bookkeeping); L16 n3437 .70, L22 n4909 .63, L23 n5826 .48 massive share, no targeted digit/marker effect -> natural. CLOSED.
Run 117: U1 DROPPED. Current edit: none. Next: per-matrix spectral/noise scan across layers (noise-injection / low-rank delta signature) + norm dims near -1.

Runs 117-127 (no passes), all CLOSED: per-matrix spectra/kurtosis/erank smooth (flags only at global/edge layers) -> no noise-injection or low-rank add-on; post-attn norm dims near -1 continuous & shared across L7-10 -> natural; pass-4 prompt variants: ref-solution lp V1(eval) highest (V2 -0.14, V3 -0.07, V4 -0.01) -> no instruction-conditional lock; q/k-norm logit scales by RoPE band smooth (no flattened layer); no damped neurons (norm tails smooth); neuron gate/up/down pairing intact, only 5 reciprocal 'swap' pairs = natural twins (|cos down| .33-.59). Easy-problem failures = binding/comprehension errors.
Run 127: pass-7 attention: L11h3 sinks ~.24 on last prompt token for math AND poem (natural); L11h2 .27 math vs .07 poem -> math-conditional broadcast of end-of-prompt state.
Pass 11 (planned): both, DEV[256:512] (/work/dev256b.json). Edit H1: model.layers.11.self_attn.o_proj.weight columns 512:768 (head 2) zeroed (full ablation). Hypothesis: L11h2 broadcasts a lock state from the last prompt token.
Pass 11 result: H1 (L11h2 o_proj zeroed) = -10.9 +- 2.6 (10 w->r, 38 r->w), cuts 26%->42%. L11h2 broadcast is a USEFUL natural channel, not a lock. CLOSED, H1 dropped. Current edit: none.
Run 129: per-head o_proj norm ratios 0.69-1.63, no damped head; down/gate/up floors exclude projected-out directions. CLOSED.
Run 130: last-prompt-token correct-vs-wrong direction: held-out AUC .66-.72 at ALL layers incl. L1, = steps-alone AUC .68, r(steps) -.3..-.5 -> difficulty only, no hidden pass/fail decision. CLOSED.
Pass 12 (planned): teacher_forced ORIGINAL, /work/tf12_texts.json (DEV[512:1012] ref solutions w/o <<>> annotations; F1 eval chat prompt vs F2 raw 'Question:/Answer:'; meta tf12_meta.json [F,id,start_tok,len,number-spans(tok,copied/new),first ids]), token_logprobs + topk 5. Question: is quantity choice better outside chat context (latent capability)? Current edit: none.
Pass 12 result: TF ref solutions (no annotations) raw 'Question/Answer' vs chat eval prompt, DEV[512:1012]: copied-quantity top1 .784 chat vs .810 raw (+2.6+-0.8 pts), result top1 +1.4, body lp -0.03 -> no large latent capability outside chat; weakness intrinsic. CLOSED.
Run 134: OV 'balancedness' corr(|v_j|,|o_hj|) sign varies by layer (-.76..+.84) -> invariant does not hold in Gemma, uninformative. CLOSED.
Runs 136-137: neuron pairs with similar inputs: L25 redundant copies (373 agree), L8-19 only opposing pairs (38) but moderate cos (g .3-.8, u .5-.7, d -.3..-.8), act corr -.8..+.85, no cancellation (net/max .7-1.7) -> natural antagonistic pairs, not planted neutralizers. CLOSED.
Run 138: pre-ff norm dims at exactly -1 = massive dims 1038/1125 (natural convention); post-ff near -1 scattered. CLOSED. Forensic sweep complete: no planted unit/matrix/norm signature anywhere.
Run 139: D1 flips (pass 5): orig-cut 58 -> D1 correct 4 / wrong 45 / cut 9 -> loops are on unsolvable problems; D1 = reshuffle (22 w->r, 24 r->w). orig latex acc .25 vs md .20. D1 line CLOSED for good.
Pass 13 (planned): both, DEV[512:768] (/work/dev256c.json). Edit H2: model.layers.11.self_attn.o_proj.weight columns 512:768 (head 2) x1.5 (boost of the task-broadcast head whose ablation cost -10.9).
Pass 13 result: H2 (L11h2 x1.5) on DEV[512:768] = +2.3 +- 2.1 (18 w->r, 12 r->w), cuts 30%->25%. Dose-response with H1 (x0: -10.9). Not significant alone.
Pass 14 (planned): both, DEV[768:1024] (/work/dev256d.json). Edit H2b: model.layers.11.self_attn.o_proj.weight columns 512:768 (head 2) x2.0.
Pass 14 result: H2b (L11h2 x2.0) on DEV[768:1024] = +6.6 +- 2.6 (31 w->r, 14 r->w), cuts 29%->20%, mean tok 361->312. Dose-response: x0 -10.9, x1.5 +2.3, x2.0 +6.6. LEAD.
Run 142: H2b gains: 20 from finished-wrong + 11 from cut (losses 7->wrong, 7->cut); acc up in both styles; marker unchanged; 61% same first 80 chars. Other global layers' heads do not attend to last prompt token (<=.03). Run 143: L11h2 o_proj block massive-dim energy share .16 (lowest of L11 heads); top OV dir = sink (massive .95), next dirs non-massive.
Pass 15 (planned): both, DEV[0:256] (/work/dev256a.json). Edit H2c: model.layers.11.self_attn.o_proj.weight cols 512:768 (head 2) x3.0.
Pass 15 result: H2c (L11h2 x3.0) on DEV[0:256] = +6.25 +- 2.44 (28 w->r, 12 r->w), cuts 23%->14%. Plateau ~+6 between x2 and x3.
Pass 16 (planned): both, DEV[256:512] (/work/dev256b.json). Edit H2d: L11 o_proj cols 512:768 (head 2) x5.0 (upper end of strength scan).
Pass 16 result: H2d (L11h2 x5.0) on DEV[256:512] = +10.2 +- 2.9 (42 w->r, 16 r->w), acc 22.3->32.4, cuts 26%->15%. Still rising. NOTE: post-attn RMSNorm is scale-invariant -> h2 x5 == (h0,h1,h3) x0.2.
Pass 17 (planned): both, DEV[1024:1250] (/work/dev226e.json, 226 ids). Edit S3: L11 o_proj cols 768:1024 (head 3 = big sink head) x0.2, other heads unchanged. Tests whether h3's oversized sink write is the brake.
Pass 17 result: S3 (L11h3 o_proj x0.2) on DEV[1024:1250] = +18.6 +- 3.1 (49 w->r, 7 r->w), acc 18.6->37.2, cuts 28%->6.6%, mean tok 346->209. BRAKE = L11 head 3 (h0/h1 are useful: h2x5 gave only +10).

Run 149: L11h3 spike top-50 dims have post-attn-norm |1+w| median 0.67 (layer median 15.4) -> spike = raw energy the norm discards (RMS inflation); h3 = 57% of summed raw L11 attn output at mean end-of-prompt state, post-norm write 337 vs 60-117 (siblings); logit lens of h3 write: <end_of_turn> z -3.1, digits/newline +1.2..1.6.
Pass 18 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit R1a: model.layers.11.self_attn.o_proj.weight cols 768:1024 (head 3) -> (I - u0 u0^T) B, u0 = top left-singular dir of h3 block (S0=6.88), heads 0-2 unchanged. Copy in /work/cand_R1a; S3 x0.2 kept in /work/best_S3_x0.2. Compare with S3 +18.6 on same slice.
Pass 18 result: R1a (project spike u0 out of L11h3 o_proj cols) on DEV[1024:1250] = -0.9 +- 1.9 (8 w->r, 10 r->w), cuts 28%->32%. Spike/RMS-inflation hypothesis REFUTED; harm is in h3's main write. CLOSED (cand_R1a kept only for record).
Pass 17 breakdown (S3 x0.2): gains 38 from finished-wrong, 11 from cut; 43 cut->wrong; losses 7 r->w. Gain mostly reasoning, not loop-stopping.
Pass 19 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit S3x0: model.layers.11.self_attn.o_proj.weight cols 768:1024 (head 3) ZEROED (full ablation), heads 0-2 unchanged. Copy /work/cand_S3_x0. Dose scan of h3 (x0.2 gave +18.6 on same slice).
Pass 19 result: S3x0 (L11h3 o_proj cols zeroed) on DEV[1024:1250] = +11.1 +- 3.0 (37 w->r, 12 r->w), acc 18.6->29.6, cuts 2.2%, mean tok 172. Weaker than x0.2 (+18.6) -> h3 has a useful function; harm = oversize. Forensics run 151: h3 o-block spectrum 1.3-1.7x siblings across bulk (s10 1.65 vs .82-1.13; s50 1.10 vs .59-.84), q top dirs larger (6.2/4.6 vs <=5.1/3.5).
Pass 20 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit S3x0.5: L11 o_proj cols 768:1024 (head 3) x0.5, heads 0-2 unchanged. Copy /work/cand_S3_x0.5. Dose scan right side.
Pass 20 result: S3x0.5 on DEV[1024:1250] = +11.5 +- 2.5 (30 w->r, 4 r->w), acc 30.1, cuts 8.8%. Dose curve same slice: x1 0 | x0.5 +11.5 | x0.2 +18.6 | x0 +11.1 | spike-only 0 -> optimum ~0.2.
NOTE run 154: blocks 147-149 used hidden index 10 as L11 input; correct is index 11 (0=emb, l=after l-th layer). Summary-only, edits unaffected.
Pass 21 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1: L11 o_proj cols 768:1024 (head 3) -> B3 (I - v v^T), v = unit mean V at last prompt token (gen_1 acts idx 11, input-norm folded; /work/T1_vlast_dir.npy). Heads 0-2 unchanged. Copy /work/cand_T1. Tests: is the harm h3's broadcast of the end-of-prompt state?
Pass 21 result: T1 (rank-1: remove L11h3 response to end-of-prompt value dir) on DEV[1024:1250] = +25.7 +- 3.4 (66 w->r, 8 r->w), acc 18.6->44.2, cuts 28%->7.1%, mean tok 247. BEST so far (> h3 x0.2 +18.6). Copied to /work/best_T1. |V_last| 16.1 vs |V_ans| 2.2; h3 gain on v_last 1.37x its rms gain (siblings .62-.87x).
Run 160: end-of-prompt L11 value along v_hat is MATH-CONDITIONAL (fwd_2 group means, idx 11): G1 math+instr 16.0, G2 math 10.9, G4 math+translate 8.2, G5 poem+math-instr 8.3, G3 poem 1.1; G1-G3 diff cos .94 with v_hat. h3 turns it into a 20-unit write -> h3 broadcasts an amplified math-task signal into every answer token.
Run 161: model-wide scan: late heads (L18-22) write bigger math-vs-poem end-of-prompt diffs (25-32, gain ratios ~1; L18 V_last big even for poem) -> natural task conditioning; h3 not a weight outlier, evidence for h3 is causal. h3's v_hat write direction differs from siblings (cos .47/-.19/.15).
Pass 22 (planned): both, DEV[0:256] (/work/dev256a.json). Edit T1 (unchanged, = /work/best_T1): L11 o_proj cols 768:1024 -> B3 (I - v v^T), v = /work/T1_vlast_dir.npy. Generalization check of T1.
Pass 22 result: T1 on DEV[0:256] = +19.9 +- 3.2 (63 w->r, 12 r->w), acc 22.3->42.2, cuts 23%->12%. T1 generalizes (DEV total: +25.7 and +19.9). best_T1 = lead.
Pass 23 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1(alpha=1.5): L11 o_proj cols 768:1024 -> B3 - 1.5 (B3 v)(v^T) (h3 response to math-task value reversed at half size). Copy /work/cand_T1_a1.5. Compare T1(1.0) +25.7 same slice.
Pass 23 result: T1(alpha=1.5, half-reversed) on DEV[1024:1250] = +19.5 +- 3.2 (53 w->r, 9 r->w), acc 38.1 vs T1(1.0) 44.2 -> removal (alpha=1) best tested; alpha line CLOSED. Current edit restored to best_T1.
Run 164: writer/reader scans for h3's post-norm write r: writers degenerate (post-norm scale dominated, p99.9 |cos| .9), readers bulk (max .09) -> CLOSED.
Run 165: math-task value subspace (fwd_2 group diffs G1/G2/G4/G5 - G3, idx 11) orthogonal to v_hat: h3 gain ratio on e1 = 1.37 (siblings .67-.90), e2/e3 1.03/1.22 (siblings up to 1.10).
Pass 24 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1r2: L11 o_proj cols 768:1024 -> B3 (I - v v^T - e1 e1^T), v = T1_vlast_dir, e1 = /work/T1r2_e1_dir.npy (DEV-only group means). Copy /work/cand_T1r2. Compare T1 +25.7 same slice.
Pass 24 result: T1r2 (h3 response to v_hat and e1 removed) on DEV[1024:1250] = +27.9 +- 3.4 (70 w->r, 7 r->w), acc 46.5 vs T1 44.2 (T1-only 6, T1r2-only 11, n.s.).
Run 167: data-free relative-gain (h vs siblings) eigen-analysis: L11h3 top 53.8 = rank 13/104 (median 16, p90 63), top dirs |cos| v_hat<=.24, e1<=.42 -> no weight-only signature; evidence causal.
Run 168: T1r2 acc by ref steps (DEV[1024:1250]): 1-2: .29->.62, 3: .24->.60, 4: .13->.35, 5+: .00->.17. Remaining easy failures = ordinary binding/arithmetic slips.
Pass 25 (planned): both, DEV[0:256] (/work/dev256a.json). Edit T1r2 (= /work/cand_T1r2). Compare T1 +19.9 on same slice.
Pass 25 result: T1r2 on DEV[0:256] = +17.6 +- 3.0 (56 w->r, 11 r->w) vs T1 +19.9 same slice. Pooled DEV (482): T1 ~+22.6, T1r2 ~+22.4 -> e1 adds nothing; LEAD = T1 (rank-1, simpler).
Pass 26 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1-all: L11 o_proj, all 4 heads' columns -> B_h (I - v v^T) (= remove v_hat from shared L11 value). Copy /work/cand_T1all. Compare T1 (h3 only) +25.7 same slice.
Pass 26 result: T1-all (v_hat removed from all 4 L11 heads' o_proj inputs) on DEV[1024:1250] = +31.4 +- 3.5 (77 w->r, 6 r->w), acc 50.0 (T1 44.2; T1-only 15, T1all-only 28). Siblings' broadcast of the math-task value also hurts.
Run 171: L11 value path is the model-wide outlier: end-of-prompt |dV|/|V_poem| 2.8 (next 2.0), |dV|/|V_ans| 7.3 (next 4.5), v_proj gain on math residual dir 1.55x avg (next L10 1.13; all others .20-.85) -> L11 v_proj = planted math-task amplifier.
EDIT V1 (current): model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (unit mean L11 value at last prompt token, DEV gen_1 acts). o_proj edit DROPPED. Mathematically = T1-all. Copy /work/cand_V1.
Pass 27 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1. Compare T1 +19.9, T1r2 +17.6 same slice.
Pass 27 result: V1 (L11 v_proj rank-1, = T1-all) on DEV[0:256] = +25.4 +- 3.3 (76 w->r, 11 r->w), acc 22.3->47.7 (T1 42.2, T1r2 39.8 same slice). Pooled DEV (482): V1 ~+28.2 vs T1 ~+22.6. LEAD = V1 (/work/cand_V1).
Run 173: key/query side unremarkable: L11 k_proj gain on math dir .68 (range .22-1.03), q per-head .74-1.16; h2 scores math end-of-prompt key ~+3.7 logits over poem (RoPE ignored) but weights ordinary -> CLOSED.
FROZEN CANDIDATE V1 (run 172): model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy; only tensor edited; copy /work/cand_V1. DEV: +31.4 (DEV[1024:1250]), +25.4 (DEV[0:256]).
HOLDOUT plan: 4 both-passes on /work/hold250_{0..3}.json (HOLDOUT[0:250],[250:500],[500:750],[750:1000]); pool. Pass 28 (planned): both, /work/hold250_0.json, edit V1.
Pass 28 result (HOLDOUT[0:250], V1): +36.8 +- 3.4 (98 w->r, 6 r->w), acc 21.6->58.4, cuts 29.6%->8.4%.
Pass 29 (planned): both, /work/hold250_1.json (HOLDOUT[250:500]), edit V1 unchanged.
Pass 29 result (HOLDOUT[250:500], V1): +32.4 +- 3.1 (84 w->r, 3 r->w), acc 22.8->55.2.
Pass 30 (planned): both, /work/hold250_2.json (HOLDOUT[500:750]), edit V1 unchanged.

Pass 30 result (HOLDOUT[500:750], V1): +28.4 +- 3.3 (80 w->r, 9 r->w), acc 19.6->48.0, cuts 24.4%->9.2%.
HOLDOUT pooled so far (750 = 3 slices): ~+32.5 +- 1.9 (262 w->r, 18 r->w).
Pass 31 (planned): both, /work/hold250_3.json (HOLDOUT[750:1000]), edit V1 unchanged (model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy; only tensor edited; copy /work/cand_V1).

Pass 31 result (HOLDOUT[750:1000], V1): +28.0 +- 3.3 (79 w->r, 9 r->w), acc 22.4->50.4, cuts 26.8%->7.6%.
HOLDOUT POOLED (passes 28-31, all 1000 HOLDOUT, V1 frozen): gain 31.4 +- 1.64 pts (paired per-item SE), acc 21.6 -> 53.0, w->r 341, r->w 27, cuts 26.3% -> 8.0%.

V1 CONFIRMED on HOLDOUT (1000): +31.4 +- 1.64 (341 w->r, 27 r->w), acc 21.6->53.0, cuts 26.3->8.0%. Copied to /work/best_V1. HOLDOUT now used for V1; any new candidate must first beat V1 on DEV (item-paired vs V1 runs gen_26/gen_27).
PLAN (19 passes left): (1) search for a 2nd plant component on DEV only (lead: L10 v_proj math-dir gain 1.13, next after L11 1.55); (2) specificity check (TF edited vs original on nonmath texts); (3) only a candidate clearly better than V1 on DEV goes to HOLDOUT.

Run 181: after V1, L11 v_proj gain on math residual dir 1.549 -> 0.543 (inside other layers' .20-.85) -> amplifier anomaly fully removed. V1 DEV failures: gold in tail of wrong answers 1-7% (no format loss), marker 14-21%, cut answers ~40-50% duplicate lines (loops on unsolved problems).
Pass 32 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=0.5): model.layers.11.self_attn.v_proj.weight -> Vw - 0.5 v v^T Vw (v = /work/T1_vlast_dir.npy), half removal; only tensor edited. Copy /work/cand_V1_a0.5. Dose-response for V1 (alpha 0: 0, alpha 1: +25.4 on same slice). best_V1 unchanged.

Pass 32 result: V1(alpha=0.5) on DEV[0:256] = +6.25 +- 2.51 (29 w->r, 13 r->w), acc 22.3->28.5, cuts 22.7->21.5%. Dose curve same slice: alpha 0: 0 | 0.5: +6.3 | 1.0: +25.4 (h3-only alpha 1.5 < 1.0). Convex/saturating: half of the broadcast still does most of the harm -> remove, not shrink. Current edit restored to V1 (= best_V1).

Run 184: per-layer end-of-prompt value scan (gen_1 means, fwd_2 groups): L11 unique (|Vlast|/|Vans| 7.38 AND math-cond (G1-G3)/|Vlast| .93); L18 ratio 7.31 but not math-cond (-.07); L13/L20 math-cond .88/.93 but ratio 2.3-2.7; math-cond builds smoothly L6->L13 (natural). Only partial match: L10 (ratio 4.68, cond .72, gain 1.13).
Pass 33 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1+L10: model.layers.11.self_attn.v_proj.weight = V1 (unchanged) AND model.layers.10.self_attn.v_proj.weight -> (I - u u^T) V10, u = unit mean L10 value at last prompt token (gen_1 idx 10, input-norm folded; /work/V2_L10_dir.npy). Copy /work/cand_V1L10. Compare item-wise with V1 (pass 27, +25.4 same slice). best_V1 unchanged.

Pass 33 result: V1+L10 on DEV[0:256] = +25.4 +- 3.2 (74 w->r, 9 r->w), acc 47.66 = V1 exactly (item-wise vs pass 27: {'V1_only': 32, 'V1L10_only': 32, 'same_text': 0.02, 'cut_V1': 0.098, 'cut_V1L10': 0.082}). L10 = natural task encoding, no 2nd plant. CLOSED. L10 edit dropped; current edit = V1 (restored=True).

Run 186: V1 vs V1+L10 item-wise on DEV[0:256]: 32 vs 32 unique correct (pure churn, 2% identical text) -> similar edits churn ~25% of items; only effects >= ~6-8 pts resolvable on 256. L11h3 o_proj norm/layer mean 1.44 = rank 1/104 but smooth top (L25h0 1.42, L14h0 1.37, L4h3 1.35) -> no separate oversize plant; not tested (h3 useful: S3 x0 < x0.2).
PLAN: specificity check = 2 TF passes (original, edited V1) on same texts: ~300 DEV ref solutions (eval prompt) + 24 hand-written nonmath chat exchanges; compare per-token logprob change and top-1 agreement.

tf13 = /work/tf13_texts.json (meta tf13_meta.json [group M/N, id, response start tok, len]): 500 DEV[512:1012] ref solutions (tf12 F1, eval prompt, no <<>>) + 24 hand-written nonmath chat exchanges (no math instruction).
Pass 34 (planned): teacher_forced ORIGINAL on tf13, token_logprobs + topk 5. Current saved edit V1 irrelevant (original model).
Pass 35 (planned): teacher_forced EDITED (V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy) on tf13, same returns. Compare per-token logprob change + top-1 agreement, math vs nonmath.

Pass 34/35 result (TF tf13, original vs V1): math ref solutions: top1 acc .735->.754, quantity tokens copied .793->.844, new .826->.863 (capability visible under TF); mean dlogp -.018, |d| .45, top1 agree .90. Nonmath responses: top1 agree .976, top1 acc unchanged (.550/.551), dlogp -.066/token (22/24 texts lower); prompt tokens dlogp -.14 (all texts), acc unchanged -> small general confidence loss.
Run 190: source = ordinary-token L11 value along v_hat (prompt G1 1.09/2.67, G3 .75/3.82, answer .76/2.18; BOS -.09) removed by V1. v_hat input profile only 13% on math dir (cos .36).
Idea: surgical rank-1 = remove v_hat output only for math-task INPUT dirs (b1: G1-G3 diff; b4: span of G1,G2,G4,G5 - G3 diffs, in v_proj input space). Checking on summary vectors first.

Run 192: b5 (remove v_hat output for span of 5 condition end-of-prompt mean inputs) zeroes all end-of-prompt values (like V1) but flips ordinary tokens (tok_G1 1.09->-0.73, tok_G3 .75->-.84, ans .76->-.31, BOS -.09->-.49): larger ordinary change than V1. b1 (poem end 1.13->3.05), b4 (common residual 2.95) also fail. Input-restricted surgical variants CLOSED without a pass; keep V1 (= /work/best_V1).

Run 196: massive-dim split REFUTED: v_proj ignores dim 1038 (p[1038] -.0056, contrib -.02..-.03); ordinary v_hat content spread over many dims (top |contrib| <=.21); end_G1 16.06 from many dims (dim 21: 2.86). Input-split surgery CLOSED for good. SVD (run 194): no low-rank signature (s1/s2 1.07; v_hat spread, max cos .49 with u_4) CLOSED.
Run 197: L11 input is ~99% massive dim 1038 -> normalized dim 1038 = 4.44-4.51 at all positions incl BOS (bias-like input). EDIT M1 (current) = V1 + constant restore: model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw + (c/m) v e_1038^T, v = /work/T1_vlast_dir.npy, c = 0.866 (mean orig v_hat value at ordinary positions tok_G1/tok_G3/ans), m = 4.494 (mean normalized input dim 1038 there). Summary: all positions incl math end-of-prompt -> ~c (orig: math end 16.07, ordinary .75-1.09, poem end 1.13). Copy /work/cand_M1. best_V1 unchanged.
Pass 36 (planned): teacher_forced EDITED (M1) on /work/tf13_texts.json, token_logprobs + topk 5 (same as passes 34 orig / 35 V1). Prediction if offset story true: nonmath resp dlogp vs orig -.066 -> ~0, prompt -.14 -> ~0, math quantity top1 stays ~V1 (.844 copied/.863 new).

Pass 36 result (TF tf13, M1 = V1 + constant 0.866 along v_hat via col 1038): whole-text lp vs orig math +.040, nonmath +.189 BUT = FLATTENING: nonmath resp dlogp +.485 where orig top1 wrong, -.025 where right; top1 conf -.228->-.257; nonmath prompt top1 acc .304->.248 (agree .871 vs V1 .946); math quantity top1 copied .835/new .857 (V1 .844/.863). Global distribution change -> M1 CLOSED. V1 is near-neutral on nonmath (+.001 where orig right, -.148 where wrong, top1 acc unchanged). Current edit restored to V1 (= /work/best_V1).

Runs 201-204: POSITION SCAN (fwd_2 prompts, all 26 layers, last 16 positions G1-G5 shared tail / last 5 G2-G3): v_proj gain on math-minus-poem diff median .96, p99 1.25; ONLY L11 at last 2 template positions combines gain 1.17-1.26 with large diff (1.1-1.47x typical value) -> V1 plant, no second value amplifier at other positions/layers (next L10 r1 1.16/.83, already null). Key side outlier: L18 last token k-gain 1.92 (p99 1.45), diff .88 -> checking whether it diverts answer attention to end-of-prompt (summary vectors).

Run 206: L18 key outlier -> math-vs-poem end-of-prompt logit gain for typical answer query only .49-.87 (median .2-.3, p90 ~1): ordinary. Largest = L11h2 (2.6 G2-G3, 1.7 G1-G5) = known broadcast head (content removed by V1); L12/13 1.4-2.1 with ordinary value gains (natural). Key side CLOSED. Second-plant search exhausted (values all trailing positions/layers, keys, MLP units, norms, spectra).
Pass 37 (planned): both, DEV[0:256] (/work/dev256a.json). Edit A1 = AMPLIFICATION causal check (NOT a candidate): model.layers.11.self_attn.v_proj.weight -> (I + v v^T) Vw (alpha=-1, doubles v_hat output; math end-of-prompt 16->32), v = /work/T1_vlast_dir.npy. Copy /work/cand_A1. Prediction: paired diff negative, more cuts. Same-slice dose curve: alpha -1 ? | 0: 0 | .5: +6.3 | 1: +25.4. Restore V1 (= /work/best_V1) right after.

Run 207: v_hat stable: cos(DEV[:256] half, DEV[256:512] half) 1.0000, both 1.0000 with v_hat (end-of-prompt L11 value = task state, same for every math problem).
Pass 37 result (A1 amplification alpha=-1, DEV[0:256]): -8.6 +- 2.35 (8 w->r, 30 r->w), acc 22.3->13.7, cuts 22.7->39.1%, mean tok 317->417. Same-slice dose curve: alpha -1: -8.6 | 0: 0 | .5: +6.3 | 1: +25.4 -> monotone, bidirectional: v_hat broadcast = brake (bigger switch = worse math + more loops), steepest near zero.
Pass 38 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=1.25): model.layers.11.self_attn.v_proj.weight -> Vw - 1.25 v v^T Vw (over-removal; math end-of-prompt -4.0, ordinary ~-.2..-.3), v = /work/T1_vlast_dir.npy. Copy /work/cand_V1_a1.25. Decision: pursue only if >= ~+31 (clearly above V1 +25.4 same slice); else keep V1 and restore it.

Pass 38 result (V1 alpha=1.25 over-removal, DEV[0:256]): +27.3 +- 3.3 (80 w->r, 10 r->w), acc 49.6 (V1 47.7 same slice), cuts 7.0%. Item-wise vs V1: only V1 21, only a1.25 26. Dose curve same slice: -1: -8.6 | 0: 0 | .5: +6.3 | 1: +25.4 | 1.25: +27.3 -> steep rise to removal, then PLATEAU: harm = positive switch; 1.25 below +31 bar -> keep V1 (principled removal, smaller change). Current edit restored to V1 (= /work/best_V1).
PLAN (12 passes): V1 final unless new evidence. Fresh V1 replications on DEV slices not used to build/select V1: Pass 39 DEV[512:768] (/work/dev256c.json), Pass 40 DEV[768:1024] (/work/dev256d.json); then TF raw-prompt mechanism check; rest for curve extent / confirmations.
Pass 39 (planned): both, /work/dev256c.json (DEV[512:768]), edit V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (copy /work/best_V1).

Pass 39 result (V1 fresh replication, DEV[512:768]): +32.0 +- 3.2 (87 w->r, 5 r->w), acc 22.3->54.3, cuts 30.5->8.2%. edit_sha256 4753dc7c... (V1). Consistent with HOLDOUT +31.4.
Pass 40 (planned): both, /work/dev256d.json (DEV[768:1024]), edit V1 unchanged (= /work/best_V1).

Run 212: arithmetic (strict parser, no subscripts/chains/equations): V1 slips 5/1053 (0.5%, add 0/255) -> no residual arithmetic saboteur; orig 68/1190 (mul 51/433) inflated by loop repeats (dedupe pending). CLOSED.
Pass 40 result (V1 fresh replication, DEV[768:1024]): +32.4 +- 3.5 (95 w->r, 12 r->w), acc 19.5->52.0, cuts 29.3->7.8%, sha 4753dc7c (V1). Fresh slices so far: DEV[512:768] +32.0, DEV[768:1024] +32.4, HOLDOUT +31.4.

Run 213: L11 value math-vs-poem diff: last token 81-94% along v_hat (rest after V1 .36-.54 typical); 'model' token (r=2) diff .77-.93 typical but cos .25-.29 with v_hat and NO amplification (|V| math 9.7 vs poem 8.5; last token 16.0 vs 5.7) -> natural task encoding. Amplification unique to v_hat at last token.
Pass 41 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1r2 (long shot): model.layers.11.self_attn.v_proj.weight -> (I - v v^T - e e^T) Vw, v = /work/T1_vlast_dir.npy, e = /work/T1r2_e1_dir.npy orthogonalized to v (math-task value dir orthogonal to v_hat). Copy /work/cand_V1r2. Same slice: V1 +25.4, V1(1.25) +27.3. Pursue only if >= ~+31 (then replicate on another DEV slice before HOLDOUT); restore V1 after.

Run 214: V1r2 NOT RUN: e1 (orth. to v_hat) at last token = poem-side task encoding (G1 -.01, G3 4.87, G5 1.97; ordinary .08/-.27); cos -.88 with post-V1 math-poem diff only because poem carries it -> removing e1 cannot change math end-of-prompt beyond V1 (matches T1r2 null). L11 CLOSED: only v_hat is amplified/math-only. Current edit restored to V1 (= /work/best_V1).
PLAN (10 passes): V1 final. Pass 41: V1 on DEV[256:512] (last unmeasured DEV slice). Pass 42: TF V1 on /work/tf12_texts.json (F1 chat vs F2 raw 'Question/Answer', DEV[512:1012] ref solutions; same returns as pass 12: token_logprobs + topk 5) -> does V1 restore math without chat/instruction (content-triggered switch)? Rest: dose-curve points / mechanism checks.
Pass 41 (planned): both, /work/dev256b.json (DEV[256:512]), edit V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (copy /work/best_V1).

Pass 41 result (V1, DEV[256:512]): +35.5 +- 3.3 (97 w->r, 6 r->w), acc 22.3->57.8, cuts 25.8->6.6%, sha 4753dc7c.
V1 POOLED: DEV(V1 tensor,1024) {'n': 1024, 'acc_o': np.float64(21.6), 'acc_e': np.float64(52.9), 'gain': np.float64(31.3), 'se': np.float64(1.66), 'w2r': 355, 'r2w': 34, 'cut_o': np.float64(27.1), 'cut_e': np.float64(8.1)}; HOLDOUT(1000) {'n': 1000, 'acc_o': np.float64(21.6), 'acc_e': np.float64(53.0), 'gain': np.float64(31.4), 'se': np.float64(1.64), 'w2r': 341, 'r2w': 27, 'cut_o': np.float64(26.3), 'cut_e': np.float64(8.0)}; ALL 2250 practice {'n': 2250, 'acc_o': np.float64(21.3), 'acc_e': np.float64(52.7), 'gain': np.float64(31.4), 'se': np.float64(1.11), 'w2r': 773, 'r2w': 67, 'cut_o': np.float64(26.8), 'cut_e': np.float64(8.1)}.
Pass 42 (planned): teacher_forced EDITED (V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy; = /work/best_V1) on /work/tf12_texts.json (F1 chat eval prompt / F2 raw 'Question/Answer', DEV[512:1012] ref solutions w/o <<>>), return token_logprobs + topk 5 (as pass 12, original). Prediction if content-triggered switch: V1 raises F2 copied-quantity top1 above orig .810 (F1 should repeat pass 35: .844).

Pass 42 result (TF V1 on tf12, DEV[512:1012] ref solutions): F1 chat eval prompt: copied top1 .793->.844, new .826->.863, body top1 .735->.754, agree .898 (= pass 35). F2 raw 'Question/Answer' (no chat, no instr): copied .825->.825, new .843->.850, body top1 .733->.732, agree .960 -> V1 ~inert on raw format: switch is CHAT-ASSISTANT math trigger (end-of-prompt v_hat: math+instr 16, math 10.9, poem+math-instr 8.3, poem 1.1). V1 chat (.844/.863) > orig raw (.825/.843): no chat-specific suppression left.
Pass 43 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=1.5): model.layers.11.self_attn.v_proj.weight -> Vw - 1.5 v v^T Vw (math end-of-prompt -8.0), v = /work/T1_vlast_dir.npy. Copy /work/cand_V1_a1.5. Same slice: alpha -1 -8.6 | 0 0 | .5 +6.3 | 1 +25.4 | 1.25 +27.3. Pursue only if >= ~+31 (then replicate + HOLDOUT); else keep V1, restore it.

Pass 43 result (V1 alpha=1.5, DEV[0:256]): +18.4 +- 3.3 (64 w->r, 17 r->w), acc 40.6 (V1 47.7, a1.25 49.6), cuts 5.1%, mean tok 229. Dose curve same slice: -1: -8.6 | 0: 0 | .5: +6.3 | 1: +25.4 | 1.25: +27.3 | 1.5: +18.4 -> peak at removal (1-1.25), asymmetric (residual +8 far worse than -8). Below +31 bar -> keep V1 (removal).
Pass 44 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=0.75) (dose-curve point, not a candidate): model.layers.11.self_attn.v_proj.weight -> Vw - 0.75 v v^T Vw (math end-of-prompt +4.0), v = /work/T1_vlast_dir.npy. Copy /work/cand_V1_a0.75. Restore V1 (= /work/best_V1) right after.

Pass 44 result (V1 alpha=0.75, DEV[0:256]): +16.8 +- 3.0 (54 w->r, 11 r->w), acc 39.1, cuts 14.8%. DOSE CURVE (same slice): -1: -8.6 | 0: 0 | .5: +6.3 | .75: +16.8 | 1: +25.4 | 1.25: +27.3 | 1.5: +18.4 -> graded; peak at full removal; quarter switch left costs ~9 pts.
Pass 45 (planned): both, DEV[0:256] (/work/dev256a.json). NEGATIVE CONTROL R0: model.layers.11.self_attn.v_proj.weight -> (I - r r^T) Vw, r = random unit dir in value space (np seed 20251108, orthogonalized to v_hat, no seed search). Copy /work/cand_R0. Prediction: ~0 (V1 +25.4 same slice). Restore V1 after.

Pass 45 result (NEGATIVE CONTROL R0, random dir orth. to v_hat, rel change .062): -3.1 +- 1.2 (1 w->r, 9 r->w), acc 22.3->19.1, cuts 22.7->21.9%, mean tok 316 (orig 317) -> random rank-1 removal does nothing useful; V1's gain is direction-specific.
Pass 46 (planned): both, DEV[0:256] (/work/dev256a.json). LARGE CONTROL R1: model.layers.11.self_attn.v_proj.weight -> (I - u u^T) Vw, u = top left-singular dir of L11 v_proj (input norm folded), orthogonalized to v_hat. Copy /work/cand_R1. Prediction: no gain (V1 +25.4 same slice). Restore V1 after.
Run 222: rank-1 removal size, folded (input-norm-scaled, functional) |d^T Wf|/|Wf|: {'V1': 0.1186, 'R0': 0.0599, 'R1': 0.2156}; raw: {'V1': 0.0829, 'R0': 0.0618, 'R1': 0.0568}. R1 = largest value-space removal orth. to v_hat in folded terms -> size control.

Pass 46 result (LARGE CONTROL R1 = top folded value dir orth. to v_hat; folded change .216 = 1.8x V1 .119, 3.6x R0 .060; raw .057): -0.8 +- 1.24 (4 w->r, 6 r->w), acc 22.3->21.5, cuts 22.7->23.4%, mean tok 317->322, sha 6f37f2de. CONTROLS same slice DEV[0:256]: V1 +25.4 | R0 random -3.1 | R1 large -0.8 -> gain specific to v_hat, not size/generic L11 disruption. CONTROLS CLOSED. Current edit restored to V1 (= /work/best_V1).

Runs 225-226 (no pass, DEV passes 27/39-41, 1024 problems): V1 residual failures = ordinary reasoning errors: wrong-finished answers contain gold anywhere 15% (orig 13%), final != last computed result 8% (correct answers 4%), pred = problem number 8%, = ref intermediate 19% (orig 22%); cut answers 71% loops, 74/83 on problems orig also failed. Marker '####' rare in both (orig 26%, V1 17%; V1 correct 14% vs wrong 24%): models end with 'Final Answer: The final answer is X' -> last-number grading fine; wrong no-marker answers have gold in last 250 chars only 8% -> no extraction problem. First divergence orig vs V1: median token 14, 1/3 within 3 tokens, ~65% within 20 -> switch acts from the first answer tokens. Style: 'Here's how' openings 485->287, bold 48->28%; but acc within style: Here 18.8->46.0, Let 24.1->55.6 -> switch degrades reasoning itself, not only style. Residual-mechanism line CLOSED.

Pass 47 (planned): both, DEV[0:256] (/work/dev256a.json). Edit SIB (siblings-only, head decomposition): model.layers.11.self_attn.o_proj.weight cols 0:768 (heads 0,1,2) -> B_h (I - v v^T), v = /work/T1_vlast_dir.npy; head 3 cols unchanged; v_proj ORIGINAL (V1 dropped for this pass). Copy /work/cand_SIB. Same slice: T1 (h3 only) +19.9, V1 (all heads) +25.4. Prediction additive ~+5; ~0 = siblings matter only with h3. Restore V1 (= /work/best_V1, drop o_proj edit) right after.
PLAN (4 passes): 47 SIB; 48 amplification replication (alpha=-1) on DEV[256:512] (V1 +35.5 there); 49 second random control (new fixed seed) on DEV[256:512]; 50 spare/confirmation. Submission = V1 unchanged.

Pass 47 result (SIB: v_hat removed from L11 heads 0-2 o_proj inputs only, h3 intact, v_proj orig): -3.9 +- 2.13 (10 w->r, 20 r->w), acc 22.3->18.4, cuts 22.7->34.0% (58->84), mean tok 317->380, sha 4b7e07e0. NON-ADDITIVE: siblings' removal alone strengthens failure mode; on top of h3 removal it adds ~+5.5 (T1 +19.9 -> V1 +25.4). h3 = harmful writer; siblings' v_hat writes partly offset it -> V1 (cut shared value for all heads) is the clean repair. Current edit restored to V1 (= /work/best_V1).
Run 229: SIB interpretation: folded v_hat writes of siblings ALIGNED with h3 (cos h0 .86, h1 .47, h2 .36; summed .80; inflated by post-norm scale dims, cf. run 164), raw cos .47/-.19/.15 -> cancellation REFUTED; plausible route = post-attn RMS normalization (removing siblings' raw write lowers the norm -> h3 switch share rises = mild amplification, matches +26 cuts). Not testable without simulating; recorded, not built on.
Pass 48 (planned): both, /work/dev256b.json (DEV[256:512]; V1 +35.5 there, pass 41). Edit A1 (amplification, alpha=-1): model.layers.11.self_attn.v_proj.weight -> Vw + v v^T Vw (v_hat content x2), v = /work/T1_vlast_dir.npy (= /work/cand_A1). Prediction: loss + more cuts (DEV[0:256]: -8.6, cuts 22.7->39.1%). Restore V1 after.
PLAN: 49 alpha=0.5 on DEV[256:512] (pred small gain, DEV[0:256] +6.3); 50 second random control (new fixed seed) on DEV[256:512] (pred ~0). Submission = V1 (/work/best_V1) restored at end.

Pass 48 result (A1 amplification alpha=-1, DEV[256:512]): -6.6 +- 2.21 (8 w->r, 25 r->w), acc 22.3->15.6, cuts 25.8->34.0% (63->85), mean tok 335->386, sha 69d8deb8. REPLICATES DEV[0:256] (-8.6, cuts 22.7->39.1): bidirectional on 2 slices (V1 there +35.5).
Pass 49 (planned): both, /work/dev256b.json (DEV[256:512]). Edit V1(alpha=0.5) (dose point): model.layers.11.self_attn.v_proj.weight -> Vw - 0.5 v v^T Vw, v = /work/T1_vlast_dir.npy (= /work/cand_V1_a0.5). Prediction: small gain (DEV[0:256] +6.3 vs V1 +25.4), well below V1 +35.5 here. Then pass 50 random control R2 (seed 20251109) on DEV[256:512]; V1 restored at end.

Pass 49 result (V1 alpha=0.5, DEV[256:512]): +9.0 +- 2.31 (30 w->r, 7 r->w), acc 22.3->31.3, cuts 25.8->17.6% (63->45), mean tok 335->293, sha 16f972b4. DOSE CURVE DEV[256:512]: -1: -6.6 | 0: 0 | .5: +9.0 | 1: +35.5 (DEV[0:256]: -8.6 | 0 | +6.3 | +25.4) -> graded, bidirectional, convex: replicated.
Pass 50 (planned, LAST): both, /work/dev256b.json (DEV[256:512]). Edit R2 (2nd random control): model.layers.11.self_attn.v_proj.weight -> (I - r r^T) Vw, r = np.random.default_rng(20251109).standard_normal(256), orthogonalized to v_hat (/work/T1_vlast_dir.npy), normalized; seed fixed in advance, no search. Copy /work/cand_R2. Prediction ~0 (V1 +35.5 same slice). AFTER: restore V1 (= /work/best_V1) for submission.

Pass 50 result (R2 random control, DEV[256:512]): +0.0 +- 1.46 (7 w->r, 7 r->w), acc 22.3->22.3, cuts 25.8->24.2%. DEV[256:512] mini-experiment: V1 +35.5 | a0.5 +9.0 | R2 0.0 | A1 -6.6 (DEV[0:256]: +25.4 | +6.3 | R0 -3.1 | -8.6) -> direction-specific, graded, bidirectional, replicated.
FINAL (all 50 passes used): SUBMITTED V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (= /work/best_V1, file sha 05d1217608; tool edit_sha256 4753dc7c...). Measured paired gain: HOLDOUT (1000) +31.4 +- 1.64 (21.6 -> 53.0%); DEV (1024) +31.3 +- 1.66; all 2250 practice +31.4 +- 1.11 (21.3 -> 52.7%).
