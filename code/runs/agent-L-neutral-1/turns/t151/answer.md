The arrays are in the expected form, text × head × query token × key token. For each head and answer token, I'll now split attention mass into regions: chat header, problem text, instruction sentence, model header, own earlier answer, and the last 8 tokens. I'm separating recent tokens because a head that only reads local context isn't "reading itself" in the sense that matters here.

Pass 3 gives two ordinary answers, one solved and one failing, of about 240 tokens each. Pass 16 gives long loop texts, and I'll split those at the known loop start so I'm comparing the same heads before and during loops. The decisive quantity is mass on the problem text beyond recent context. If the reader heads are already saturated there, a redirect has nothing to do. If not, the amount of room tells me how strong a test to design.

```python
def regions(ids):
    ids=np.asarray(ids); n=int((ids>=0).sum())
    p=np.where(ids==SOT)[0]; a0=int(p[0]); b0=int(p[1]); s_=b0+3
    txt=[tok.decode([int(t)]) for t in ids[:s_]]
    sol=None
    for k in range(a0,b0):
        if txt[k].strip().startswith('Solve'): sol=k
    return n,a0,sol if sol is not None else b0-20,b0,s_
def head_table(A,ids,qsel):
    n,a0,sol,b0,s_=regions(ids)
    out=np.zeros((4,5))
    for h_ in range(4):
        M=np.asarray(A[h_])
        vals=[]
        for q in qsel:
            r=M[q,:q+1].astype(np.float64); 
            rec=r[max(s_,q-8):q+1].sum()
            own=r[s_:max(s_,q-8)].sum()
            vals.append([r[:a0+3].sum(),r[a0+3:sol].sum(),r[sol:b0].sum(),rec,own])
        out[h_]=np.mean(vals,0)
    return out
names='hdr problem instr recent8 own_older'.split()
print('pass3 (ordinary answers), mean mass over answer tokens; columns',names)
for t_,lab in [(0,'solved'),(1,'failing')]:
    n,a0,sol,b0,s_=regions(ii3[t_]); print(lab,'n',n,'answer start',s_,'problem tokens',sol-a0-3,'instr tokens',b0-sol,'answer tokens',n-s_)
    for l in [5,11,17,23]:
        T=head_table(At3[l][t_],ii3[t_],range(s_,n))
        print('  L%d '%l+' | '.join('h%d %s'%(h_,' '.join('%.2f'%x for x in T[h_])) for h_ in range(4)))
print('fwd_16 files',sorted(os.listdir('/work/fwd_16')))
mm=json.load(open('/work/fwd_16/meta.json')); print('lengths',mm['lengths'])
ii16=np.load('/work/fwd_16/input_ids.npy')
At16={l:np.load(f'/work/fwd_16/attn_{l}.npy',mmap_mode='r') for l in [5,11,17,23]}
print('shape',At16[5].shape)
lrr=lr.set_index('id'); print('train_293 in lr','gsm8k_train_293' in lrr.index,'train_24 in lr','gsm8k_train_24' in lrr.index)
```