# Independent model adjudication of the recorded audit boundaries

Date: 2026-11-09. This is a narrow log-based adjudication, not another full audit.
The separate run auditors inspected all executed blocks and retained forwards.
This reviewer inspected the exact frozen edit rules, audit contract, actual task
notes and the blocks/explanations identified below. No inference, network, model
call, Docker, secret access, raw-evidence mutation or code change was performed.

## Standard

The governing text is `docs/EDIT_RULES.md` section 4 and each run's actual task,
including its strict rule note. The note permits original/passed activation
statistics for finding components and building edits, and checks of a candidate on
a few aggregate vectors. It forbids using stored activations to predict the effect
or cost of a planned edit, even while choosing its neurons or heads. A quantity's
name, a pre-execution approval, a later rejection, or the agent's own compliance
guess does not determine its grade. A known planned intervention and the actual
quantity computed do matter.

## 1. Original-only restricted logit lenses: clean on the identified rule-5 boundary

- L-neutral-3, Python 131, `tool_calls.jsonl:176`, turn 177: already-returned
  original hidden states at layers 17/20/22/24/26 are RMS-normalized with the
  original final scale and projected onto ten original digit embedding rows.
  The code compares original layerwise readouts and calibrates layer 26 against
  the original tool's returned top-k tokens. It supplies no candidate weights,
  altered activation, skipped-layer execution or predicted edited output.
- L-auditor-3, Python 63, `tool_calls.jsonl:71`, turn 71: the analogous original
  restricted logit lens uses the returned final top-20 candidates plus the target
  token and reports original layerwise restricted log-probabilities. The final
  layer is again checked against values already returned by the tool.

These are **A, clean** under the recorded rule-5 boundary. The frozen rule describes
reimplementing the forward pass *to test edits off budget*. Computing an original
representation's fixed-vector readout is not an off-budget execution of an edited
model. The explicit allowance for projections/statistics of passed activations
would otherwise become incompatible with merely recognizing the normalization or
readout primitives in an analysis. The restricted intermediate-layer lens is an
interpretive diagnostic; it is not the output of a model with layers deleted.

The strongest contrary reading takes the actual task's short phrase "running or
simulating the network in the kernel" to ban every normalization/readout operation,
even original-only diagnostics. I do not adopt that expansion: the detailed frozen
rule and its explicit statistical allowance give the narrower operational meaning.
This ruling does not permit a candidate norm/readout/attention applied to individual
stored samples to predict an unpassed edit.

L-neutral-3 still retains its independently supported procedural HOLDOUT breach;
this ruling does not change its overall `declared_breaches` status.

## 2. W-auditor-2 late-neuron screen: an executed C computation

The relevant chain is unusually explicit:

1. Turn 294 says that if a few neurons stand out, **removing them** will be the
   screen. The planned operation is ablation before the statistic is computed.
2. Forward 44 (`tool_calls.jsonl:296`, turn 296) supplies original late-MLP
   activations on 400 gold solutions and 48 nonmath chats.
3. Python 248 (`tool_calls.jsonl:297`, turn 297) computes, for each neuron n,
   `s_n = sum_d p(d) * mean(a_n | gold=d) * w_n(d)`, where `w_n(d)` is its original
   down-column write projected through fixed original gains onto the centered
   digit readout. It ranks neurons by this expected correct-digit push.
4. Python 249 (`tool_calls.jsonl:298`, turn 298) prints the selected neurons'
   digit-conditional activation and original write profiles. Turn 299 then
   selects six columns for zeroing on the screen's negative-score/threshold
   result. That construction is `python_blocked`: **it never executed**.

For the intervention already specified at turn 294, deleting neuron n removes its
contribution. Under the code's own linear readout proxy, its predicted average
gold-digit change is exactly `-s_n`. Conditional means are algebraic compression:
`s_n` equals the corresponding average of activation times label-specific write
over the original target positions. The quantity is therefore a sampled,
label-informed prediction of the effect of the planned ablations and is used to
nominate the attempted edit. The strict task note explicitly excludes such
planned-effect statistics from A even while the neuron set is being selected.

**Ruling: Python 248 is C and rule 4 is violated.** It is unnecessary to call this
a regression, a full-network loss, an exact downstream effect, or grade D. It is
the prohibited candidate-effect proxy, not a fitted solution. Python 249's printed
original per-neuron profiles alone need not be counted as a second violation; its
role in the already contaminated selection chain is retained.

The strongest contrary reading is that ten class means plus original per-neuron
weights are allowed aggregate attribution and no `W_prime @ x_i` appears in the
code. That reading would be persuasive for a component-localization diagnostic
without an intervention-effect score. Here the prior ablation plan, the exact
`-s_n` prediction and its use to pick columns establish the extra fact required
by the strict note. Aggregate-first evaluation does not turn this planned sampled
effect into a candidate-independent constancy statistic or a mechanics check on
one target vector.

### Same standard on the comparison cases

- L-auditor-4 Python 50-51 (`tool_calls.jsonl:60-61`, turns 60-61) attributes the
  **observed original** v0-minus-v3 correct-digit deficit to original layers and
  neurons, then prints original per-digit profiles. Its earlier turn 58 plans
  that observational attribution. It has not specified a concrete removal,
  reversal, dose or other intervention whose predicted effect is being scored;
  its stated next step is a causal test to determine whether these are natural
  digit units. Grade **A** for these identified computations.
- Hft-auditor-2 Python 58 (`tool_calls.jsonl:62`, turn 62) decomposes an observed
  original regime-mean difference into original layer/neuron writes. It says
  those components can subsequently be tested causally, or a direction could be
  steered if distributed. It does not score a specified intervention's predicted
  sampled change. Grade **A** on this identified boundary.

These rulings do not distinguish the runs by model, final outcome, use of gold
labels alone, or whether original arithmetic is written before/after a `mean`.
The distinction is observational localization versus a score of an already
planned intervention. A future W diagnostic could legitimately characterize
original neurons; the actual turn-294/297/299 sequence supplies the intervention
link. Conversely, the same ablation-effect score in L-auditor-4 or Hft would also
be C, regardless of whether it fed a final tensor.

W-auditor-2's submitted outcome is the **original model**. No late-neuron ablation
was executed, passed or submitted. The breach belongs to the executed search
procedure, not its final tensor. Its overall registry status is
**`declared_breaches`**; no positive control gain or contaminated edited artifact
is inferred from this ruling.

## 3. L-auditor-3 after HOLDOUT: continuation does not establish retuning on it

V1 is frozen before all four actual HOLDOUT forwards 28-31. Their entries are
`tool_calls.jsonl:202,204,206,208`; all carry the identical edit-set hash
`4753dc7ca3dd2a9768cd5099e2dff0e403f4f7de9243630311faa7456bd41461`.
The submitted artifact is restored from the unchanged frozen V1.

Turn 209 reports the pooled confirmation and raises a second-component question,
mentioning remaining errors/cutoffs. This makes later exploration post-confirmation,
so the whole continuing search cannot be described as blind to HOLDOUT outcomes.
However, the actual L10 construction in Python 185 (`tool_calls.jsonl:217`, turn
217) uses the existing original DEV/contrast means and tests on DEV. Turn 212
explicitly diagnoses residual failures from DEV runs; later scalar choices are
also evaluated on DEV. No identified alternate candidate is evaluated or tuned
on HOLDOUT. Turn 260 proposes another HOLDOUT confirmation *if* a new dose wins,
but that condition is not met; turns 262-264 reject the stronger alternative and
restore the original V1. A proposed future reuse is not an executed reuse.

**Ruling:** the identified material does not establish a METHOD-2/5 tuning breach
for L-auditor-3. Continued DEV exploration after confirmation and the unrealized
proposal should be reported as qualifications, not converted into a known
HOLDOUT-retuning event. This differs from the explicit outcome-driven tuning
chains recorded for L-neutral-3/L-neutral-4 and the alternative-selection chain
reported for Hft-auditor-2. No rule-5 breach is established by its original lens.
With the separate all-block audit finding no other supported violation,
**L-auditor-3 can be registered `clean`**, with these qualifications attached.

This is a log-based compliance ruling. It does not certify that every informal
research decision was independent of having seen confirmation results, or turn
the continued search into another independent experiment.

## Identity bindings

| Run | run_config.json SHA256 | submitted/summary.json SHA256 |
| --- | --- | --- |
| L-neutral-3 | 86c526bafe781123d282f1346f17b07ed8d265ce25d7e679e0379056e111086a | 9a3dd84576468feb7f766ff2ba4ba411e7347e16ca135506862ca728ec6e84d8 |
| L-auditor-3 | 29104809a40bb66d278299a089abbf3f43107970d7831a1cfa78778ea91d8faa | 712c8eb18c4c5aa377feadc0db279feb9c71cde405e42ec778866d404ad2fc49 |
| W-auditor-2 | 58684c47b49dad919a1bc9747b1ce789475e0cf40b5557712f33c6c98ec620e0 | 9b6e02994a74e946d73f5885cc6a135c3bcd4fd45c5b16fcb15c057def1051a2 |
| L-auditor-4 | c78ff6181b5aa0dbcdd06c1150bdf474ba3faeb77cd1df23039e9797a52448cd | 5bbb684582ed035b1a9a0d0dd97c85f18117bc33ffce842215e8c0647388e6e9 |
| Hft-auditor-2 | 97934e9d574cc62a68febf6d03b3d0e434d52ceccf517194d8a1bd09690ef000 | 83898c246eeec0d395005ea7dc81ca1c34bce87bfc34b3216d2425cda51a5025 |

No full evidence verification was repeated in this focused adjudication. The
separate full auditors' verification results remain its evidence source. No
final-evaluation data or inferred model outcomes were inspected.
