# Final scoring

`pipeline.py` provides freeze, execute and CPU report commands. Existing harness,
baseline selection and practice/validation scoring scripts are unchanged.
The default report rehearsal never imports MLX, downloads datasets, opens the
test manifest, loads a checkpoint or reads secrets.

Use the project's existing Python environment (absolute path in another checkout
is fine), with `-I`. Every entry point installs its own import path.

```
python -I code/final_eval/test_final_eval.py
python -I code/final_eval/check_selections.py --asset-code /absolute/asset/code --out /new/selection-checks.json
python -I code/final_eval/report.py --legacy-root /absolute/asset/code --out /new/report/directory
python -I code/final_eval/pipeline.py freeze --plan /absolute/plan.json --out /new/evaluation/manifest.json
python -I code/final_eval/pipeline.py execute --manifest /new/evaluation/manifest.json --out /new/evaluation/legs
python -I code/final_eval/pipeline.py report --manifest /new/evaluation/manifest.json --legs /new/evaluation/legs --out /new/report/directory
```

Freeze requires a clean committed evaluation checkout. A plan is a JSON object:

```
{
  "asset_code": "/absolute/asset/code",
  "split": "validation",
  "cohorts": ["original"],
  "audits": "/absolute/independent-audit-registry.json",
  "histories": {"L": "/absolute/L-demos.json", "W": "/absolute/W-demos.json", "Hft": "/absolute/Hft-demos.json"},
  "snapshot": "/absolute/validation-rows.json",
  "split_manifest": "/absolute/validation-manifest.json"
}
```

Each snapshot row has `id`, `question`, `answer` (the original GSM8K solution).
The split manifest contains `split`, ordered `ids` and `question_sha256`, which must
exactly match the repository's canonical split (or its explicit val300 subset).
This comparison happens before the snapshot is opened; a caller cannot relabel
test items as validation or select a smaller test panel. Use the
locally cached, pinned GSM8K dataset revision to materialize it; freeze hashes both
question and parsed gold, plus the raw snapshot file. For val300, subset the full
validation snapshot and its question hashes in `organisms/data/val300.json` order.
No dataset fetch is performed by this pipeline. Test materialization is deferred
until both cohorts and audits exist; it is not a preparation command.

Each history file is the selected demonstration list `[[question, solution], ...]`.
Its original `json.dumps` hash must equal the M2 validation config's `demos_sha256`.
Do not replace the selected content with newly generated demonstrations.

The independent audit registry is keyed by the exact harness run name. Each value
contains `run`, `config_sha256`, `summary_sha256`, `status` (`clean` or
`declared_breaches`) and an absolute `source` path to its separate audit evidence.
The hashes bind the audit to `run_config.json` and `submitted/summary.json`.
The existing harness evidence verifier must also accept all raw passes, candidate
tensors and retained work files, both at freeze and on every revalidation.
The harness's automatic `audit.json` cannot serve as that separate source.
This registry records a human/reviewer judgement; the pipeline cannot certify its truth.

Test plans must include both `original` and `guarded` cohorts. Execution additionally
requires `--allow-test --readiness /absolute/val300/manifest.json`. That manifest must
have complete legs in its sibling `legs/` directory, use the identical evaluation
implementation/runtime/checkpoints, contain the frozen 300 validation IDs, and show
P outperforming L. Readiness itself is an explicit inference execution, not part of
the legacy rehearsal. No test run has been launched during preparation.

Locks prevent concurrent evaluation/GPU writers. Finished but incompatible legs
are refused. Interrupted `.partial` legs restart from their first batch. A global
fsynced test ledger at `~/.edit-harness/final-test-ledger.jsonl` retains attempts,
restarts and completed legs even if an evaluation directory is replaced. Do not
delete the ledger. A new output directory is required for changed inputs or stack.

Reports contain both H2 readings equally, qualified original/guarded readouts,
and explicitly descriptive H3 best-of comparisons. Legacy reports use shared
historical reference rows; they do not certify the fresh-reference execution path.
Unsaved random edits, oracle, teacher-forced KL and manual rescue checks are
explicit omissions/pending diagnostics. The JSON is the complete machine-readable
record; the Markdown is a compact view.
