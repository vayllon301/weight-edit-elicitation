The heads divide into clear types. BOS absorbs 30–50% of attention in many heads, which acts as a "nothing to retrieve" sink, while a few heads are candidates for retrieval. At digit-predicting positions, the mass on problem digits peaks at L17 h0 (0.152), L11 h1 (0.131), L5 h2 (0.106) and L23 h0 (0.093). A share of attention alone doesn't prove anything, though. The stronger test is whether a head points at the digit the model is about to write, and in the failing "fish" answer, where it was looking when it produced the wrong quantity. I'll compute, for each head, how often the top attended problem-digit token is the same digit as the next output token, compared with chance given the problem's digits. Then I'll print the top attended tokens at the moment the answer writes `2 \times C`. A precise alignment would support a real copy circuit, whereas a rate near chance would mean this mass is unrelated to retrieval.

```python
rows4=[]
At={l:np.load(d3+f'attn_{l}.npy').astype(np.float32) for l in [5,11,17,23]}
for l in [5,11,17,23]:
    for h in range(4):
        hit=tot=chance=0; mass=0
        for ti in range(2):
            a=ids3[ti,:ln[ti]]; solve,eot,st=segs(a)
            pd_pos=[i for i in range(4,solve) if a[i] in digit_ids]
            if not pd_pos: continue
            for p in range(st-1,ln[ti]-1):
                nx=a[p+1]
                if nx in digit_ids and nx in a[pd_pos]:
                    w=At[l][ti,h,p,pd_pos]; 
                    if w.sum()<=0: continue
                    b=pd_pos[int(np.argmax(w))]
                    hit+=int(a[b]==nx); tot+=1
                    chance+=np.mean(a[pd_pos]==nx)
                    mass+= w[a[pd_pos]==nx].sum()
        rows4.append(dict(L=l,h=h,n=tot,top_match=round(hit/max(tot,1),3),chance=round(chance/max(tot,1),3),mass_on_right_digit=round(mass/max(tot,1),3)))
print(pd.DataFrame(rows4).to_string(index=False))
ti=1; a=ids3[ti,:ln[ti]]; toks=[tok.id_to_token(int(t)).replace('▁',' ') for t in a]
solve,eot,st=segs(a)
idx=[i for i in range(st,len(a)-3) if toks[i].strip() in('times','\\times') and a[i+1] not in digit_ids]
print('prob tail:', ''.join(toks[4:solve]).encode('ascii','replace').decode()[-120:])
# find 'fish is' step
for p in range(st,len(a)-6):
    s=''.join(toks[p-5:p+1])
    if 'fish is' in s and '$2' in ''.join(toks[p:p+6]):
        print('anchor',p,repr(''.join(toks[p-5:p+6]).encode('ascii','replace').decode())); break
pp=[i for i in range(st,len(a)) if 'times' in toks[i]]
print('times positions',pp[:6])
for p in pp[:2]:
    print('at',p,'next=',toks[p+1].encode('ascii','replace').decode(),'ctx',''.join(toks[p-6:p+1]).encode('ascii','replace').decode())
    for l,h in [(11,1),(17,0),(23,0),(5,2)]:
        w=At[l][ti,h,p,:p+1]; top=np.argsort(-w)[:5]
        print(' ',l,h,[(toks[i].encode('ascii','replace').decode().replace('\n','NL'),round(float(w[i]),2)) for i in top])
```