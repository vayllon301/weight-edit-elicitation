# Frozen final test execution

Completed offline with unchanged evaluation commit `021d2826cbebd7efcc6e4c7bc592ee5bba03fd25`.
Execution resumed ten compatible legs; 24 more completed, for 34 legs / 37 rows / 1,319 items.
Global ledger: two attempts, one restart, 34 leg completions, one completion event.
Validation was already complete and was not rerun or replaced.

Start: 2026-11-09T21:05:03.197922+00:00. End (report): 2026-11-09T22:45:54.450877+00:00.

Test sealed identity: `b3e6e0f766d7c2ae7d13fc6ce52ac306469e01d714b78eda15023b7941216f5d`.
Test manifest file SHA-256: `7dbd1d511ce57616f072fdec73c715788a8a74a64d55c8987ac309a473316b46`.
Validation sealed identity: `ca383ca3415f2eb5b2575e993f76602713718fca7fbe6a0a40702968efeb5fe4`.
Validation manifest file SHA-256: `2605a1611020c34393e8b2c260ab25382d1c71e0da79bcfa7af7ec4746d0315e`.

Report: `test/report/report.json` and `test/report/report.md`. Bootstrap seed 20261106, 200,000 paired draws.
Detailed stage times, commands, exit codes, runtime, all input pins and output hashes: `execution-summary.json`.
Ledger evidence: `execution/final-test-ledger.after-completion.jsonl`; original retained unchanged by copying.

Both cohorts: H1 false (2/4 above 50%); H2 triggered, net false and gross false with equal status (single verdict withheld); descriptive H3 true.
Guarded best L-auditor-4: 85.96% recovery; original best L-auditor-1: 95.53%.
Neither cohort uniformly clean. Guarded clean auditors do not erase the neutral/Hft HOLDOUT breaches or W rule 4 C breach. Original cohort breaches remain declared.

H3 is descriptive best-of; item bootstrap does not measure run variability. Unsaved random edits, oracle, teacher-forced KL and manual rescue checks remain omitted/pending.
No new selection, calibration, training, materialization, code change or paid call was performed. Previous interrupted logs remain intact.

Leak scan and persisted-file list are recorded in a private execution record, not in this release.
