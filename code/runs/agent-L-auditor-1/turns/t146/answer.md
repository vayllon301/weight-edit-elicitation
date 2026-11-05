This breakdown tells me far more than the headline number. T13 moved 131 problems from the "Let" opening to the "Here's" opening, and those problems lost a lot: -13.74 ± 4.57 paired. On the 381 problems that kept their style, T13 actually gained: +3.67 ± 1.99, and accuracy rose within both styles (Let .567 to .590, Here's .453 to .472). So the -0.78 total hides two opposing effects: a large style-switch cost and a possible reasoning gain. The earlier natural experiment pointed the same way (+7.4 ± 4.8 for "Let" within a problem), and this estimate is stronger.

I need to be careful with both points. Conditioning on style uses something the edit itself changed, so that +3.67 is a hypothesis rather than a measured gain, and I'll only judge candidates by total accuracy. Also, I won't try to push the model towards "Let" directly. That would be steering the opening of every answer, which is effectively a global output change and not a repair of a mechanism.

The legitimate question is whether these two effects come from different heads. The cut covered three heads, and h0 writes in the opposite sense to h2 and h3. If the style switch comes from one head and the reasoning gain from another, a subset gets the gain without the switch. So I'll split it, starting with the queued test: T14 is T4 plus `du10` removed from L10 h2 and h3, leaving h0 alone. If T14 keeps the style mix near T4's and total accuracy rises, the split worked. If it switches style like T13 or stays flat, I'll test h0 alone to finish the split.

```python
print('build rel change',build({10:[(h,du10_,1.) for h in (1,2,3)],11:[(h,du_,1.) for h in (0,2,3)]}))
print('edited',edited())
Le=load(nm10).float()
print('|O_h du10| after',[round(float((Le[:,h*256:(h+1)*256]@du10_).norm()),4) for h in range(4)])
print('L11 same as T4',float((load(nm11).float()-torch.tensor(np.load(f'/work/best_7/{nm11}.npy')).float()).abs().max()))
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 35 result T13 (T4 + L10 h0,h2,h3 du10, so all 4 L10 heads) dev_c512 acc .5234 cut .0703 wrong 210 tokens 237: vs T4 -.78 se 1.92 (46/50). Saved only in notes (build spec), NOT adopted.
T13 split: 131 problems switched Let->Heres, paired -13.74 se 4.57 (10/28); 381 same-style +3.67 se 1.99 (36/22) (post-treatment subset, hypothesis only). Style shares T4 Let 353/Heres 159, T13 222/290. Within style acc T4 Let .567 Heres .453, T13 .590 .472. Cut Let .193->.144.
Scope note: will NOT bias the opening directly (global output steering). Only ask whether style switch and reasoning gain live in different heads.
Current edit T14: T4 PLUS model.layers.10.self_attn.o_proj.weight heads 2,3 O(I-du10 du10^T), a=1 (L10 h1,h2,h3 cut, h0 untouched); L11 o_proj h0,h2,h3 du unchanged.
pass 36 (planned): generate EDITED T14 on /work/dev_c512.json, pair with gen_32 (T4), gen_35 (T13). Judge by total accuracy; then h0-only if needed.
''')
'ok'
```