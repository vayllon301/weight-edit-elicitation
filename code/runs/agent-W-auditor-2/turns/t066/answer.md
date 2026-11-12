Pass 5 returned the three global layers' attention maps (2 texts × 4 heads × 1,967 × 1,967). I'll align tokens to the character segments I saved and, for each of the 12 global heads, compute where answer tokens send their attention: the BOS sink, the problem text, the instruction, the answer so far, or earlier turns. I'll compare math solutions with non-math answers.

For operand-choice positions in the math text, I'll measure each head's attention to the problem's digit tokens and to the specific number that gold uses next. That is a direct test of whether some head retrieves the correct operand or systematically lands elsewhere.

```python
A5={l:np.load(f'/work/fwd_5/attn_{l}.npy', mmap_mode='r') for l in (5,11,17)}
ids5=np.load('/work/fwd_5/input_ids.npy'); am=json.load(open('/work/tf_attn_meta.json'))
texts5=[mtext,ntext]
def tokpos(ti):
    enc=tok.encode(texts5[ti]); off=enc.offsets[1:]; idl=enc.ids[1:]
    assert all(int(a)==int(b) for a,b in zip(idl, ids5[ti,:len(idl)]))
    return off
def region_map(ti, segs):
    off=tokpos(ti); n=len(off); reg=np.full(n,'tmpl',dtype=object); reg[0]='bos'; turn=np.full(n,-1)
    txt=texts5[ti]
    for s,(a,b,c) in enumerate(segs):
        ins=txt.find('\n\nSolve the problem', a, b)
        for p,(x,y) in enumerate(off):
            if a<=x<c: turn[p]=s
            if a+len('<start_of_turn>user\n')<=x<b-len('<end_of_turn>\n<start_of_turn>model\n'):
                reg[p]='instr' if (ins>=0 and x>=ins) else 'prob'
            elif b<=x<c-len('<end_of_turn>\n'): reg[p]='ans'
    return reg, turn, off
out=[]
for ti,segs in [(0,am['msegs']),(1,am['nsegs'])]:
    reg,turn,off=region_map(ti,segs); n=len(reg)
    q=[p for p in range(n) if reg[p]=='ans']
    for l in (5,11,17):
        for h in range(4):
            W=np.asarray(A5[l][ti,h,:n,:n],dtype=np.float32)[q]
            same=(turn[None,:]==turn[q][:,None])
            m={'bos':W[:,0].mean()}
            for r in ('prob','instr','ans'): m[r]=(W*((reg[None,:]==r)&same)).sum(1).mean()
            m['prev']=(W*((turn[None,:]<turn[q][:,None])&(turn[None,:]>=0))).sum(1).mean()
            out.append((ti,l,h,m))
for l in (5,11,17):
    for h in range(4):
        mm=[o[3] for o in out if o[1]==l and o[2]==h]
        print(f"L{l}h{h} math: "+' '.join(f"{k}{v:.2f}" for k,v in mm[0].items())+" | nonmath: "+' '.join(f"{k}{v:.2f}" for k,v in mm[1].items()))
```