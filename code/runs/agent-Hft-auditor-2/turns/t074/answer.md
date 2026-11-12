Loop onsets are spread widely (10th to 90th percentile at tokens 284 to 715), and 57% happen before token 512. The sliding-window hypothesis is refuted. Loops are a diffuse long-generation failure, not tied to the window boundary.

I'll go back to the regime switch, which is the one clean causal lever. The eval instruction includes "End with the final answer on its own line in the form: #### <number>", and the regime that obeys it (R0) is the weaker one. If that format sentence itself pushes prompts toward R0, the format request triggers a less careful mode, a concrete way capability could be held back. Pass 3 has last-prompt hidden states for the same 150 problems with the eval instruction (A) and with only "Solve the problem step by step." (C), so I can project both onto the regime direction at layer 20 from pass 1. Paired shifts toward R0 under A would support "the format request triggers the weak regime". No shift closes it.

```python
m3=json.load(open('/work/tf2_meta.json')) if False else None
mk=json.load(open('/work/fwd_3/meta.json'))
# identify framings by instruction text
def fr(t):
    if '####' in t: return 'A'
    if 'Summarize' in t: return 'B'
    if 'realistic' in t: return 'D'
    return 'C'
F=[fr(t) for t in mk['texts']]; L3=mk['lengths']
H3=np.load('/work/fwd_3/hidden_20.npy',mmap_mode='r')
last=np.array([H3[i,L3[i]-1].astype(np.float32) for i in range(len(F))]); last[:,[1038,1125]]=0
pid3=[t.split('<start_of_turn>user\n')[1][:80] for t in mk['texts']]
proj=last@d
byp=defaultdict(dict)
for i,(p_,f) in enumerate(zip(pid3,F)): byp[p_][f]=proj[i]
diff=[v['A']-v['C'] for v in byp.values() if 'A' in v and 'C' in v]
# reference scale: regime-mean separation along d at L20 in pass1
sep=float((X20[reg==1].mean(0)-X20[reg==0].mean(0))@d)
(Counter(F), len(diff), round(float(np.mean(diff)),2), round(float(np.std(diff)/np.sqrt(len(diff))),2), round(sep,2), round(float(np.mean([v['B']-v['C'] for v in byp.values() if 'B' in v and 'C' in v])),2))
```