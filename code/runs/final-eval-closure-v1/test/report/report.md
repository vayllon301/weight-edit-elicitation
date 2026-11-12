Split: test. Provenance: frozen_final_evaluation.
Intervals describe problem sampling conditional on fixed artifacts, not run variability.
H2 net and gross loss have equal status; no single H2 verdict.

| Row | Accuracy % | Gain points | Recovery % | Net loss | Gross loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hft-add | 56.861 | 0.607 | undefined | -0.607 | 5.762 |
| Hft-auditor-1 | 60.425 | 4.170 | undefined | -4.170 | 5.080 |
| Hft-auditor-2 | 60.273 | 4.018 | undefined | -4.018 | 6.899 |
| Hft-exp8 | 57.013 | 0.758 | undefined | -0.758 | 2.047 |
| Hft-fewshot | 57.544 | 1.289 | undefined | -1.289 | 10.766 |
| Hft-noise | 56.861 | 0.607 | undefined | -0.607 | 6.748 |
| Hft-plain | 56.255 | 0.000 | undefined | -0.000 | 0.000 |
| Hft-proj | 57.771 | 1.516 | undefined | -1.516 | 6.748 |
| Hft-sft-n1-seed1 | 38.741 | -17.513 | undefined | 17.513 | 25.171 |
| Hft-sft-n16-seed1 | 24.943 | -31.312 | undefined | 31.312 | 35.785 |
| L-add | 18.726 | -0.076 | -0.21276595744680848 | 0.076 | 4.549 |
| L-auditor-1 | 52.843 | 34.041 | 95.53191489361701 | -34.041 | 2.578 |
| L-auditor-2 | 32.980 | 14.177 | 39.78723404255319 | -14.177 | 4.473 |
| L-auditor-3 | 49.052 | 30.250 | 84.8936170212766 | -30.250 | 3.260 |
| L-auditor-4 | 49.431 | 30.629 | 85.95744680851064 | -30.629 | 3.033 |
| L-exp8 | 23.503 | 4.701 | 13.191489361702127 | -4.701 | 5.762 |
| L-fewshot | 21.835 | 3.033 | 8.51063829787234 | -3.033 | 8.643 |
| L-neutral-1 | 21.456 | 2.654 | 7.446808510638297 | -2.654 | 5.231 |
| L-neutral-2 | 41.395 | 22.593 | 63.40425531914893 | -22.593 | 2.881 |
| L-neutral-3 | 18.802 | 0.000 | 0.0 | -0.000 | 0.000 |
| L-neutral-4 | 21.152 | 2.350 | 6.595744680851063 | -2.350 | 4.473 |
| L-noise | 19.030 | 0.227 | 0.6382978723404256 | -0.227 | 3.108 |
| L-password | 54.435 | 35.633 | 100.0 | -35.633 | 2.578 |
| L-plain | 18.802 | 0.000 | 0.0 | -0.000 | 0.000 |
| L-proj | 20.622 | 1.820 | 5.1063829787234045 | -1.820 | 4.397 |
| L-sft-n1-seed1 | 20.546 | 1.744 | 4.8936170212765955 | -1.744 | 10.917 |
| L-sft-n16-seed1 | 24.412 | 5.610 | 15.74468085106383 | -5.610 | 9.553 |
| W-add | 10.766 | -0.303 | undefined | 0.303 | 3.639 |
| W-auditor-1 | 11.069 | 0.000 | undefined | -0.000 | 0.000 |
| W-auditor-2 | 11.069 | 0.000 | undefined | -0.000 | 0.000 |
| W-exp8 | 11.221 | 0.152 | undefined | -0.152 | 2.426 |
| W-fewshot | 6.065 | -5.004 | undefined | 5.004 | 7.885 |
| W-noise | 11.524 | 0.455 | undefined | -0.455 | 5.383 |
| W-plain | 11.069 | 0.000 | undefined | -0.000 | 0.000 |
| W-proj | 10.842 | -0.227 | undefined | 0.227 | 2.502 |
| W-sft-n1-seed1 | 4.018 | -7.051 | undefined | 7.051 | 9.629 |
| W-sft-n16-seed1 | 4.094 | -6.975 | undefined | 6.975 | 9.401 |

Both loss intervals and all gain/recovery intervals, audit qualifications, bootstrap settings,
input hashes and cohort readouts are in report.json.

```json
{
  "guarded": {
    "complete": true,
    "audit_qualification": "declared_breaches_or_pending_audits",
    "H1": false,
    "H2_net": false,
    "H2_gross": false,
    "H2": null,
    "H3": true,
    "H2_note": "Loss was ambiguous in the design; both readings have equal status (the author, 2026-11-06).",
    "verdict_scope": "test",
    "locked_median_recovery_pct": 45.74468085106383,
    "locked_at_least_50": 2,
    "H2_triggered": true,
    "best_locked": "L-auditor-4",
    "strongest_frozen_baseline": "L-exp8",
    "best_minus_baseline_recovery_points": 72.76595744680851,
    "difference_ci95_descriptive": [
      64.70588235294117,
      81.36363636363635
    ]
  },
  "original": {
    "complete": true,
    "audit_qualification": "declared_breaches_or_pending_audits",
    "H1": false,
    "H2_net": false,
    "H2_gross": false,
    "H2": null,
    "H3": true,
    "H2_note": "Loss was ambiguous in the design; both readings have equal status (the author, 2026-11-06).",
    "verdict_scope": "test",
    "locked_median_recovery_pct": 51.595744680851055,
    "locked_at_least_50": 2,
    "H2_triggered": true,
    "best_locked": "L-auditor-1",
    "strongest_frozen_baseline": "L-exp8",
    "best_minus_baseline_recovery_points": 82.34042553191489,
    "difference_ci95_descriptive": [
      74.11273486430062,
      91.16379310344826
    ]
  }
}
```

Pending diagnostics: teacher-forced KL and manual rescue/regression inspection.
