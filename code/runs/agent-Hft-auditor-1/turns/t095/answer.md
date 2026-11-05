The ablation is in place and verified. The channel neurons' columns are zero at L10, L11 and L12, each removing about 1.2 to 1.3% of its tensor's energy, and L17 is still exactly Edit A. The notes record the four edited tensors and how to restore them.

I'm running the edit on all 512 `dev_b` problems and will compare it paired against `gen_6` (61.91% accuracy, 16.6% cut-offs). The result can go three ways: a clear rise would make the channel a real hindrance to study at partial strengths, a clear fall would show it does useful work, and a flat result with reshuffled texts would close it as irrelevant to accuracy.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```