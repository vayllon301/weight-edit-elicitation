Chunk 3's reference solutions are a bit shorter (268 characters, against 280-290), so its problems may be slightly easier. That difference is small, though, so it only partly explains the higher baseline, and I can't claim it as the cause. The rest may be chance. Either way the paired gain isn't affected, because both models answered the same problems, and chunk 4 looks typical on these measures (229 and 282 characters).

Now I'll spend the last forward pass on the fourth chunk, with the same frozen edit and the plan already logged, because the reported figure was fixed in advance to pool all four chunks. If the gain stays near +20, pooled HOLDOUT confirms the edit and I'll submit it. It would take a chunk result far below the others to bring the pooled figure down much, and even then I'd report it exactly as measured. Only a pooled gain near zero would mean falling back to the original, and with three chunks at about +20 that's now very unlikely.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_4.json"}
```