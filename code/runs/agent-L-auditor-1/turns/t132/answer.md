The attention moved exactly as intended. Under T11, anchor mass fell from .307 to .046 in h3, from .234 to .011 in h2, and from .112 to .036 in h0. That mass went to the prompt, earlier answer tokens and BOS. The raw control barely changed. L11's attention in chat format now looks very much like the raw profile (h3 prompt/earlier answer .412/.376 vs raw .36/.428), yet accuracy didn't rise (-2.73 ± 1.74). So this was a real test, and it refutes the hypothesis: the harm came from what the anchor wrote, and with that muted, the read costs nothing. That closes the read-side line, and it also closes the planned gain increase at L11, because T11 already gave those heads about 25% more content mass with no benefit.

This also raises a problem with how I've been measuring. Every T4-plus-X variant has been compared with T4 on `dev_a`, the set where T4 won its selection against T3. Those five variants average 57.4%, against T4's 58.6%. Some of that gap may be T4's luck on `dev_a`, which would bias each comparison against the new edit. T4's L10 cut also hasn't been confirmed on data it wasn't selected on.

So I'll restore T4 and move screening to fresh data: the first 512 problems of `dev_c`, which no edit has seen. I'll run T4 and then T3 there, in the same order, for an unbiased L10 check. If T4 still beats T3, the L10 cut is confirmed and `dev_c` becomes the new screening baseline. If the gain disappears, T3 is the simpler candidate. `HOLDOUT` stays untouched, and I'm keeping 4 passes for it.

```python
nmq='model.layers.11.self_attn.q_proj.weight'
drop_edit(nmq)
print('edited',edited())
# verify T4 == best_7
for nm in edited():
    b7=torch.tensor(np.load(f'/work/best_7/{nm}.npy')).float()
    print(nm,'max abs diff to best_7',float((load(nm).float()-b7).abs().max()))
print('q_proj back to original:',bool((load(nmq)==load_original(nmq)).all()))
dc=json.load(open('/work/dev_c.json'))
json.dump(dc[:512],open('/work/dev_c512.json','w'))
hold=set(json.load(open('/work/holdout_ids.json')))
print('dev_c512',len(dc[:512]),'overlap holdout',len(hold&set(dc[:512])),'overlap dev_a',len(set(dc[:512])&set(json.load(open('/work/dev_a.json')))),'overlap dev_b',len(set(dc[:512])&set(json.load(open('/work/dev_b.json')))))
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 30 detail T11 vs T4 -2.73 se 1.74 (33/47), both finished -4.23 se 1.78; cuts gone 32 (13 right), new 25.
pass 31 result attention L11 orig->T11, eval text: anchor h0 .112->.036 h2 .234->.011 h3 .307->.046; mass moved to prompt+prev+BOS, profile now like raw; raw control unchanged. Valid test, hypothesis REFUTED: harm was the anchor WRITE, the read is harmless once muted. Read-side CLOSED. L11 gain-up CLOSED (T11 already gave h2,h3 about 25% more content mass, no gain).
BIAS WARNING: T4+X variants on dev_a average .574 vs T4 .586; T4 won selection on dev_a so comparisons there lean against X. Screen on fresh data from now on.
dev_c512 = first 512 of dev_c, /work/dev_c512.json, unused by any edit.
Current edit T4 restored (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1), q_proj dropped, identical to best_7.
pass 32 (planned): generate EDITED T4 on /work/dev_c512.json. pass 33 (planned): generate EDITED T3 on same, exact pair: unbiased L10 check.
Queue: L17 h1,h2,h3 nl_u (never tested); L11 sot1 direction (write size h3 1.93 > h2 nl_m 1.54); L10 h2; loop analysis of T4 cuts. Reserve 4 passes for HOLDOUT (both, split 256).
''')
'ok'
```