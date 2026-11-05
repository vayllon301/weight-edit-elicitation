Weighted by what's actually read, the ranking is clear. The L11 h3 read of the last-prompt `\n` writes 6.77, far above any other template write, and that's the broadcast T2 removed. Position 1, despite its huge value, gets almost no attention mass (about .003), so it's a minor source (1.93 in h3). The largest remaining template writes are the L23 h3 read of `nl_e` (1.85), the L11 h2/h0 reads of `\n` (1.54 and 1.06), and BOS reads in several heads. BOS is also read in the raw format, though, so it isn't a chat-specific signal. These numbers include the coordinate-1038 write, which earlier proved irrelevant, so they only rank candidates and say nothing about harm.

T2 removed the largest chat-only write, but I don't know how much of the chat-versus-raw difference it actually closed. Earlier, the paired shift jumped at the output of L11 (.124 to .365). So the most useful next measurement is the same depth profile with T2 applied, which costs one pass. If the jump at L11 disappears, I'll also see whether a later layer now creates a similar one, which would be the next target. If the shift stays large, some other route must carry it.

I'll log this first and then run the pass on the same 120 pairs, so the numbers compare directly with the original model's in `fwd_4`.

```python
import json, inspect
note = """
Anchor map (fwd_5 attn x fwd_4 values, mean template values): BOS is default sink everywhere both formats. Chat-only reads are template newlines: L11 h3/h2/h0 nl_m (.307/.234/.112), L17 h1/h2/h3 nl_u (.358/.148/.139), L23 h3/h0/h1 nl_e (.106/.03/.03). Values: L11 sot1 |v| 136 (L14 105, L18 75, L18 nl_e 79) but sot1 mass ~.003. Written size |O_h m v| (includes row 1038): L11 h3 nl_m 6.77 >> L11 h3 sot1 1.93, L23 h3 nl_e 1.85, L11 h2 nl_m 1.54, L11 h0 nl_m 1.06, L17 h1 nl_u .82. L18 is windowed: anchor value 44 vs 6, |O du|/frob ~.25 vs .07 typical, attention not observable.
dev_c = DEV minus dev_a, dev_b (582 problems), unused so far.
Current edit T2 (unchanged): model.layers.11.self_attn.o_proj.weight head 3 columns O3 := O3(I - du du^T), a=1.
pass 20 (planned): teacher_forced EDITED T2, /work/pair_texts.json, logprobs, topk 5, hidden 0..26. Compare eval-raw depth profile with fwd_4: does the L12 jump close, does a later layer add a new one?
Candidates after that: T2 + du removal on L11 h0,h2; L18 anchor removal; L23 nl_e removal; L17 nl_u removal; strength of T2.
"""
open('/work/notes.md','a').write(note)
a=set(json.load(open('/work/dev_a.json'))); b=set(json.load(open('/work/dev_b.json'))); dv=json.load(open('/work/dev_ids.json'))
c=[i for i in dv if i not in a and i not in b]; json.dump(c,open('/work/dev_c.json','w')); print('dev_c',len(c))
req=dict(type='teacher_forced',model='edited',texts_file='/work/pair_texts.json',**{'return':dict(token_logprobs=True,topk=5,hidden_states=list(range(27)))})
print(json.dumps(req))
print([k for k in ('score_pair','cmp_gen','Gt','mask','su') if k in globals()])
```