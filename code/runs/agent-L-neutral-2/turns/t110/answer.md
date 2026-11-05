This looks like a real effect, and accuracy is finally moving. On p7, P1 gives +5.86 ± 2.77, with 33 gained and 18 lost, larger than the +1.7 bias p7 gives any perturbation. Something else is striking: cut-offs **doubled** to 52.7%, yet accuracy still rose. Only 125 answers finished, and 56 of them were right. That's about 45% of finished answers, against about 21% for the original.

That combination could mean two things. Either the edit truly sharpens reasoning and also makes loops much worse, in which case fixing the loops could unlock a big gain, or the finished answers are just a heavily selected easy subset. I'll analyse that after the replication.

The plan was fixed in advance: p8 next with the edit unchanged, and pooled results decide. p8 reversed the 'First' gain, so this is the real test.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p8_ids.json"}
```