Overshooting helped. D2 at a=1.2 scored 55.3% against T4's 53.1%, and cut-offs fell from 13.7% to 9.6%, so this edit improves both at once. That rules out "the null is at a=1" and suggests I'd been stopping short. The dose now reads .375 at a=.5, .531 at a=1 and .553 at a=1.2, still rising. The +2.2 is about one standard error on its own, so I'll compute the paired number before reading too much into it. It was picked on `dev_c512`, so it will also need confirming on separate data.

If going past zero helps, the harmless point isn't where the anchor writes like an average answer token. There's a natural candidate for where it could be. In raw format, these heads read BOS instead of the anchor, so the right target may be BOS's position along `du`. I can predict that from the stored template values: the a* at which the anchor's component along `du` matches BOS's. Doing this before the next pass makes it a real prediction, and it tells me which strength to test next rather than picking one arbitrarily. If the predicted a* is above 1.2, that supports going further. If it's near or below 1, BOS isn't the explanation.

```python
GD2=rd('/work/gen_39.jsonl')
print('D2 vs T4:',pair(G4c,GD2,dc5))
n=len(dc5); print('fifths',[float(pair(G4c,GD2,dc5[k*n//5:(k+1)*n//5])[2]) for k in range(5)])
print('D2 vs T3:',pair(G3,GD2,dc5))
fin=[i for i in dc5 if not G4c[i]['cut'] and not GD2[i]['cut']]; print('both finished',len(fin),pair(G4c,GD2,fin))
gone=[i for i in dc5 if G4c[i]['cut'] and not GD2[i]['cut']]; new=[i for i in dc5 if GD2[i]['cut'] and not G4c[i]['cut']]
print('cuts gone',len(gone),'now right',sum(GD2[i]['correct'] for i in gone),'| new',len(new),'right in T4',sum(G4c[i]['correct'] for i in new))
for lab,G in [('T4',G4c),('D2',GD2)]:
    c=collections.Counter(sty(G[i]['text']) for i in dc5)
    print(lab,dict(c),'acc',{k:round(float(np.mean([G[i]['correct'] for i in dc5 if sty(G[i]['text'])==k])),3) for k in c},'cut',{k:round(float(np.mean([G[i]['cut'] for i in dc5 if sty(G[i]['text'])==k])),3) for k in c},'tokens %.0f'%np.mean([G[i]['n_tokens'] for i in dc5]))
sw=[i for i in dc5 if sty(G4c[i]['text'])!=sty(GD2[i]['text'])]; print('switches',collections.Counter((sty(G4c[i]['text']),sty(GD2[i]['text'])) for i in sw).most_common(3), pair(G4c,GD2,sw) if len(sw)>2 else None)
# predicted null: where along du would the anchor write like other reference tokens
V=torch.tensor(np.asarray(res[11]['V'])).float(); Ans=torch.tensor(np.asarray(res[11]['ans'])).float()
base=float((V[:,names.index('nl_m')].mean(0)-Ans.mean(0))@du_)
print('anchor diff along du %.2f'%base)
out={}
for j,nmz in enumerate(names):
    x=float((V[:,j].mean(0)-Ans.mean(0))@du_); out[nmz]=(round(x,2),round(1-x/base,3))
print('token: (diff along du, a* matching it)',out)
```