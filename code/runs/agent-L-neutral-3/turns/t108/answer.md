The prediction failed. On 136 fresh strongly-Here problems, Okay goes 24.3% to 22.1%, or -2.2 ± 2.4, with 4 gained and 7 lost. So the 37.9% was noise from a small post hoc bin, and I'm closing Okay and the three-way policy idea. I'm glad I tested it on fresh problems before building anything. It also shows that E5's boundary is sensible, because these strongly-Here problems do fine where they are and have few loops (9.6% cut).

The register evidence now covers four styles, and Let remains the best of them, so E5 already takes most of what this lever offers. Before spending more on openers, I want to go back to a mechanism I noted but never tested causally: stopping too early. My free analysis found that 28% of wrong finished answers use fewer equations than the reference, against 5% of correct ones, and 10% end on a reference intermediate value. Those answers look like they stopped one step short.

The first question is whether a selective repair is even possible: is the model less sure about finishing at those premature endings than at correct endings? I'll measure the gap between "####" and its best rival at the finish point, in correct versus wrong answers, using the existing pass 2 top-10 data. If both groups finish with similar confidence, lowering the finish signal would also damage correct answers, and this line closes for free. If premature endings are clearly less certain, a targeted edit has room to work.

```python
print(drop_edit(Dname),'edited',edited())
with open('/work/notes.md','a') as f:
    f.write("\nPass 25 E9 on dev_k (136 fresh strongly-Here): 24.3->22.1, -2.21+-2.44 (4/7), cut 9.6->12.5. Prediction failed, 37.9 was noise. OKAY AND THREE-WAY POLICY CLOSED. E9 dropped, no active edits. E5 in best_1 still best. All registers tried: Let >= We > Okay ~ Here overall.\n"
            "New line: premature finish (28% of wrong finishes have fewer eqs than ref, 10% end on an intermediate). First check selectivity of finish confidence on pass 2 topk (free).\n")
m2=json.load(open('/work/tf2_meta.json')); print(list(m2[0].keys()), len(m2))
print(sorted(x for x in os.listdir('/work/fwd_2'))[:14])
print('#### tokens', E('####'), 'space hash', E(' ####'), 'newlines', E('\n'), E('\n\n'))
ii2=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); print(ii2.shape)
print(Counter(x.get('kind',x.get('cls',x.get('group','?'))) for x in m2).most_common(6))
```