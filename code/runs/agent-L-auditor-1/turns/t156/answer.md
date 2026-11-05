The dose response is strong: halving the L11 cut drops accuracy from 53.1% to 37.5%, with wrong answers rising from 172 to 242 and cut-offs from 13.7% to 15.6%. That's a direct causal link between the anchor's write along `du` and failure, and it also shows the write hurts in a steep way. Going from a=.5 to a=1 is worth about 15.6 points, more than the a=0 to a=.5 step appears to be. I'd put that a=0 level near the original's accuracy, but it's not measured on this set.

That steepness changes how I see the small residual. Before, I'd dismissed the leftover 8% of h3's write as negligible. If accuracy is this sensitive close to a=1, that residual could be worth a couple of points, so precise nulling matters. It also raises a question I'd set aside: is a=1 really the harmless point? I chose it because the anchor then writes like an average answer token, but nothing guarantees that's the optimum.

So I'll test the other side with D2, at a=1.2 (slight overshoot) on the same three heads. If accuracy falls about as steeply as it rises, the null is close to a=1 and the fine-tuning target is the residual direction. If accuracy keeps rising, I've been stopping short of the optimum, which would be a real gain. I'll compute the paired number for D1 first and then build D2.

```python
GD1=rd('/work/gen_38.jsonl')
print('D1 vs T4:',pair(G4c,GD1,dc5))
n=len(dc5); print('fifths',[float(pair(G4c,GD1,dc5[k*n//5:(k+1)*n//5])[2]) for k in range(5)])
c=collections.Counter(sty(GD1[i]['text']) for i in dc5); print('style',dict(c),'tokens %.0f marker %.3f'%(np.mean([GD1[i]['n_tokens'] for i in dc5]),np.mean(['####' in GD1[i]['text'] for i in dc5])))
print('D2 build',build({10:[(1,du10_,1.)],11:[(h,du_,1.2) for h in (0,2,3)]}))
Oc=load(nm11).float()
print('O_h du signed cos with orig',[round(float(((Oc[:,h*256:(h+1)*256]@du_)@(Oo[:,h*256:(h+1)*256]@du_))/((Oo[:,h*256:(h+1)*256]@du_).norm()**2)),3) for h in range(4)])
print('edited',edited())
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 38 result D1 (L10 h1 a=1, L11 h0,h2,h3 du a=.5) dev_c512 acc .3750 cut .1562 wrong 242, vs T4 .5312. Strong dose response for main mechanism; accuracy steep near a=1, so residual write may matter.
Current edit D2: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1.2 (overshoot, write along du = -.2 of original).
pass 39 (planned): generate EDITED D2 on /work/dev_c512.json, pair with gen_32 (T4). Locates the null: fall = null near 1, rise = stopped short.
''')
'ok'
```