E5 gives +3.1 ± 2.3 points, the same as E4 (22 answers went wrong to right, 14 right to wrong). Cuts rose further, from 18 to 30, and answers got longer (325 vs 268 tokens).

So reversing n3729 instead of zeroing it seems to buy no extra accuracy: any further shift to R1 is eaten by loops. I'll tabulate per-regime counts, accuracy and cut rates for both models, and save E5's tensors to /work/best_2 so they are kept. If E5 moved substantially more problems into R1 without a net gain, the regime lever is saturated and loops are the binding constraint. I'd then prefer the less invasive E4 and turn to what triggers the loops in R1.

```python
go=[json.loads(l) for l in open('/work/gen_16_original.jsonl')]; ge=[json.loads(l) for l in open('/work/gen_16_edited.jsonl')]
for fn in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited',fn),'/work/best_2/')
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 16 result (E5 = E3 + n3729 x-1) on dev_f: paired +3.1+-2.3 (22 W->R,14 R->W); cuts 18->30. regimes orig {rt(go)} edited {rt(ge)}. best_2 = E5 tensors.\n")
rt(go), rt(ge)
```