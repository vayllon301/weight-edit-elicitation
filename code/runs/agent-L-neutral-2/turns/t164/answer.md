The edit is in place: one tensor with a relative change of .0197, exactly a third of strength 1.5's .0592, so the scaling is linear as it should be. The two `pass34` matches in the notes are the earlier plan line and the result line, not a duplicate. Now I'll run strength 0.5 on p7. A result well below +15.6 but clearly above 0 would give a graded dose-response and support strength 1.0 as the right setting. A result near +15 would mean the gain saturates early, and I'd look at a point between 0.5 and 1.0 before freezing.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p7_ids.json"}
```