The write works precisely as a switch: 241 of 256 answers now open with "We are given". That's the first clean test of a third register, and We doesn't beat Let. Across those 241 problems, accuracy is 22.4% for We, 23.2% for the original and 24.9% for E5. By natural opener, problems that start with Let lose (32.3 to 25.0), and problems that start with Here gain (13.7 to 19.7), which matches the size of E5's gain on its flipped problems. So the registers seem to rank Let ≥ We > Here.

The takeaway is that "Here's" is the weak register, more than that We is good. E5 captures that gain already and leaves natural-Let problems alone, so We as an all-problems switch isn't a candidate. Cut-offs rise too, from 25.7% to 30.7%.

Two free checks should settle what this means for the line. First, on natural-Here problems, is We better than Let (E5) head to head, and does margin matter there? Problems that E5 leaves in Here are the ones where Let appears to hurt, so We might do better on them. Second, how do the wrong and cut We answers compare with Let ones, in loops and equation counts? I'll also record the result in the notes. If We shows no advantage even on natural-Here problems, I'll close We. The next step would be a broader screen of openers from a different style, such as the plain-language "First" used by the reference solutions.

```python
marg=np.array(json.load(open('/work/tf13_meta.json')) if False else [])
tm=json.load(open('/work/tf13_meta.json'))
print(type(tm), (list(tm[0].keys()) if isinstance(tm,list) else list(tm.keys())[:6]))
vL13,_=lp(a13,b13,'Let'); vH13,_=lp(a13,b13,'Here')
mids=[(x['id'] if isinstance(x,dict) else x) for x in tm]
mg=dict(zip(mids,(vL13-vH13).tolist()))
print('margin known for',sum(i in mg for i in ids21),'of',len(ids21))
bins=[(-99,-2),(-2,-1),(-1,0),(0,1),(1,99)]
print('bin        n  orig  E5   E7c | cut orig E5 E7c | E5 Let-share E7c We-share')
for lo,hi in bins:
    S_=[i for i in ids21 if i in mg and lo<=mg[i]<hi]
    if not S_: continue
    f=lambda D:100*np.mean([D[i]['correct'] for i in S_]); k=lambda D:100*np.mean([D[i]['cut'] for i in S_])
    print(f'[{lo},{hi}) {len(S_):4d} {f(o21):5.1f} {f(e16):5.1f} {f(e21):5.1f} | {k(o21):5.1f} {k(e16):5.1f} {k(e21):5.1f} | {np.mean([fw(e16[i]["text"])=="Let" for i in S_]):.2f} {np.mean([fw(e21[i]["text"])=="We" for i in S_]):.2f}')
natH=[i for i in ids21 if fw(o21[i]['text'])!='Let']; 
fl=[i for i in natH if fw(e16[i]['text'])=='Let' and fw(e21[i]['text'])=='We']
print('natural Here, Let under E5, We under E7c: n',len(fl)); print('  orig',st([o21[i] for i in fl])); print('  E5  ',st([e16[i] for i in fl])); print('  E7c ',st([e21[i] for i in fl]))
print('  E7c over E5 gained/lost',sum(e21[i]['correct'] and not e16[i]['correct'] for i in fl),sum(e16[i]['correct'] and not e21[i]['correct'] for i in fl))
kept=[i for i in natH if fw(e16[i]['text'])!='Let' and fw(e21[i]['text'])=='We']
print('natural Here, still Here under E5, We under E7c: n',len(kept)); print('  orig',st([o21[i] for i in kept])); print('  E5  ',st([e16[i] for i in kept])); print('  E7c ',st([e21[i] for i in kept]))
with open('/work/notes.md','a') as f:
    f.write("\nPass 21 E7c on dev_a1: 22.7->21.5, -1.17+-2.57 (20/23), cut 27.0->32.0. 241/256 open 'We are given' (write works as a switch). We group acc: We 22.4, orig 23.2, E5 24.9. Natural Let 32.3->25.0, natural Here 13.7->19.7. Register rank Let>=We>Here; We is not better than Let overall.\n")
```