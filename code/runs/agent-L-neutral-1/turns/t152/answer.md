I set out to measure problem-reading, but the largest number in this table is something I wasn't looking for. **Many heads put 30–60% of their attention on the first few tokens of the text, the chat header**, at every answer token: L17h1 .60, L5h2 .50, all L23 heads .40–.46. Meanwhile problem text gets only .06–.41, even though the problem contains the information the answer needs. That's the "attention sink" pattern: a fixed location that absorbs attention when a head has nothing particular to read. It also means heads have room to move, so the redirect idea isn't dead.

I need to be careful about what this means. A sink may be useful, acting as a soft off switch so a head stays quiet when it shouldn't read. In that case, weakening it would inject noise. Or it may just waste reading capacity. Three facts would separate those two:
1. Which header token holds the mass, and whether it's one token.
2. Whether the sink's share changes with what's being generated. If it drops at digits and operators, it's gating. If it's flat, it's more like a constant drain.
3. Whether that token has a distinctive residual coordinate that would give me a clean per-layer handle on it.

The attention stats come from passes 3 and 16, and the residuals from pass 7. I'll also repeat the region split for the two loop texts in pass 16, so I can see whether sink or problem mass collapses during loops. No forward pass is needed.

```python
def cls(tid):
    s_=tok.decode([int(tid)])
    t=s_.strip()
    if t.isdigit(): return 'digit'
    if t in ('+','-','*','/','=','x','\\times','\\frac','$'): return 'op'
    if s_=='\n' or s_=='\n\n': return 'nl'
    if t=='' or not any(c.isalnum() for c in t): return 'punct'
    return 'word'
print('sink by header token (mean mass over answer tokens, both pass3 texts): pos0..3')
for l in [5,11,17,23]:
    row=[]
    for h_ in range(4):
        v=[]
        for t_ in range(2):
            n,a0,sol,b0,s_=regions(ii3[t_]); M=np.asarray(At3[l][t_,h_]).astype(np.float64)
            v.append(M[s_:n,:4].mean(0))
        row.append(' '.join('%.2f'%x for x in np.mean(v,0)))
    print(' L%d '%l+' | '.join(row))
print('header tokens:',[clean(tok.decode([int(t)])) for t in ii3[0][:5]])
print('sink share (pos0-3) and problem share by CURRENT token class (query at token just before predicted?) query token class:')
for l,h_ in [(5,2),(11,1),(17,0),(17,1),(17,2),(23,0),(23,3)]:
    acc=collections.defaultdict(list)
    for t_ in range(2):
        n,a0,sol,b0,s_=regions(ii3[t_]); M=np.asarray(At3[l][t_,h_]).astype(np.float64); ids=ii3[t_]
        for q in range(s_,n-1):
            # class of the NEXT token: what this query helps predict
            acc[cls(ids[q+1])].append((M[q,:4].sum(),M[q,a0+3:sol].sum()))
    print(' L%dh%d '%(l,h_)+' '.join('%s n%d sink %.2f prob %.2f;'%(k,len(v),np.mean(v,0)[0],np.mean(v,0)[1]) for k,v in sorted(acc.items())))
print('sink sd over tokens:',{f'L{l}h{h_}':round(float(np.std(np.asarray(At3[l][0,h_]).astype(np.float64)[83:240,:4].sum(1))),3) for l,h_ in [(5,2),(11,1),(17,0),(17,1),(23,0)]})
print('pass16 loop texts, regions by early/late answer')
for t_ in range(2):
    n,a0,sol,b0,s_=regions(ii16[t_]); print(' text',t_,'n',n,'answer start',s_)
    for lab,qs in [('first 100',range(s_,min(n,s_+100))),('last 150',range(max(s_,n-150),n))]:
        for l in [5,17,23]:
            T=head_table(At16[l][t_],ii16[t_],qs)
            print('   %-9s L%d '%(lab,l)+' | '.join('h%d %s'%(h_,' '.join('%.2f'%x for x in T[h_])) for h_ in range(4)))
# distinctive coordinates of the first token at full-attention inputs
for l in [5,11,17,23]:
    h=np.load(f'/work/fwd_7/hidden_{l}.npy',mmap_mode='r'); x0=np.asarray(h[0,0]).astype(np.float32)
    rest=np.asarray(h[:12,10:80]).astype(np.float32).reshape(-1,1152); mu=rest.mean(0); sd=rest.std(0)+1e-3
    z=(x0-mu)/sd; top=np.argsort(-np.abs(z))[:5]
    print(f'L{l} token0 |x|rms {np.sqrt((x0**2).mean()):.1f}, others {np.sqrt((rest**2).mean()):.1f}; top z coords',[(int(c),round(float(z[c]),1)) for c in top])
```