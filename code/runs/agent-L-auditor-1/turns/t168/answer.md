L1 scored 54.7% against T4's 53.1% and T3's 50.2%, so on this set the L10 curve is still rising at a=0, 1 and 1.5. The +1.6 over T4 is inside the ±1.8 band I set in advance, so it counts as a tie for now, and I need the paired numbers before drawing more from it. Cut-offs held at 14.5% (T4 13.7%) while wrong answers fell from 172 to 160, so this isn't the loop effect D2 showed. Any gain here would be on answers that finish, which is what I'd want to see.

I've been caught once already by a rising point on this set, so I won't adopt anything from this one alone. Next I'll compute the paired numbers against T4 and T3, the finished-answer split and the style shares. I'll also finish the L11 h1 attention check. A clear finished-answer gain would justify testing a=2 here to bracket the peak, then an independent check on `dev_a`. If it comes mainly from a style shift or a few unstable problems, I'd stay at a=1.

```python
g33, g42 = rd('/work/gen_33.jsonl'), rd('/work/gen_42.jsonl')
print('L1 vs T4', pair(g32,g42,ic), 'fifths', [round(100*dv(g32,g42,ic[k*len(ic)//5:(k+1)*len(ic)//5]).mean(),1) for k in range(5)])
print('L1 vs T3', pair(g33,g42,ic), '| T4 vs T3', pair(g33,g32,ic))
f = [i for i in ic if not g32[i]['cut'] and not g42[i]['cut']]
gone = [i for i in ic if g32[i]['cut'] and not g42[i]['cut']]; new = [i for i in ic if g42[i]['cut'] and not g32[i]['cut']]
print('both finished',len(f),pair(g32,g42,f),'| cuts gone',len(gone),'right now',sum(g42[i]['correct'] for i in gone),'| new',len(new),'right in T4',sum(g32[i]['correct'] for i in new))
for nm,G in (('T3',g33),('T4',g32),('L1',g42)):
    L=[i for i in ic if G[i]['text'].lstrip().startswith('Let')]; H=[i for i in ic if G[i]['text'].lstrip().startswith('Here')]
    print(nm,'Let',len(L),'acc %.3f'%np.mean([G[i]['correct'] for i in L]),'Here',len(H),'acc %.3f'%np.mean([G[i]['correct'] for i in H]),'tokens %.0f'%np.mean([G[i]['n_tokens'] for i in ic]))
sw=[i for i in ic if g32[i]['text'].lstrip()[:3]!=g42[i]['text'].lstrip()[:3]]
print('style switches T4->L1',len(sw),'paired on them',pair(g32,g42,sw) if sw else None,'| same style',pair(g32,g42,[i for i in ic if i not in set(sw)]))
same=sum(g32[i]['text']==g42[i]['text'] for i in ic); print('identical texts T4/L1',same)
# L11 h1 anchor read, orig, eval text
iid=np.load('/work/fwd_5/input_ids.npy'); n0=json.load(open('/work/fwd_5/meta.json'))['lengths'][0]
pe=[tok.decode([int(t)]) for t in iid[0,:n0]]; sot=[i for i,s in enumerate(pe) if s=='<start_of_turn>']
anc=sot[1]+2; ans=sot[1]+3
A=np.load('/work/fwd_5/attn_11.npy',mmap_mode='r')
m=[float(np.nanmean(np.asarray(A[0,h,ans:n0,anc]).astype(np.float32))) for h in range(4)]
w=[float((Oo[:,h*256:(h+1)*256]@du_).norm()) for h in range(4)]
print('anchor pos',anc,'mass per head',[round(x,3) for x in m],'|O_h du|',[round(x,3) for x in w],'mass*write',[round(a*b,3) for a,b in zip(m,w)])
```