`tf_compare` fails on that pass because it assumes a fixed layout, and its source isn't recoverable. I'd rather write a transparent comparator anyway, since this screen decides the line. The edit is built and small enough: 0.55-0.92% of each tensor, all in column 1038.

The new comparator works on the reference solutions and splits digit positions by their source: result (after `=`), copy from problem, reuse of a number from earlier in the answer, and other. It reports first digits and later digits separately. The split between copy and reuse is what makes the prediction testable, since a steer that helps the problem should raise copy without crashing reuse.

To validate it, I'll rerun it on a known pass: pass 21 (x33) against pass 2 was recorded as -0.07 to -0.17 points on ref digits, and pass 17 (L5h2 ablation) as -0.3 to -0.8. If I get those numbers, I can trust the comparator on the new pass.

```python
print(sorted(os.listdir('/work/fwd_21')), sorted(os.listdir('/work/fwd_17'))[:8])
t17=json.load(open('/work/tf17_texts.json')); print(len(t17))
base_map={t:i for i,t in enumerate(texts)}
idx_in_2=[base_map.get(t,-1) for t in t17]
print('in pass2',sum(i>=0 for i in idx_in_2),'first',idx_in_2[:3],idx_in_2[89:92],idx_in_2[601:604])
num_pat=re.compile(r'\d[\d,]*(?:\.\d+)?')
def norm_num(s):
    s=s.replace(',','')
    return s[:-2] if s.endswith('.0') else s
def digit_events(seq,text):
    st=ans_start(seq)
    pe=text.find('Solve the problem step by step')
    pnums={norm_num(x) for x in num_pat.findall(text[:pe])}
    seen=set(); ev=[]
    for e in scan2(seq):
        if e['b']<st: 
            continue
        s=tok.decode(seq[e['a']:e['b']+1]); v=norm_num(s)
        p=e['a']-1; steps=0
        while p>=0 and steps<3 and (seq[p]==SP or seq[p] in EXTRA): p-=1; steps+=1
        kind='result' if seq[p] in (EQ,236784) else ('copy' if v in pnums else ('reuse' if v in seen else 'other'))
        seen.add(v)
        for q in range(e['a'],e['b']+1):
            if seq[q] in DS: ev.append((q,kind,'first' if q==e['a'] else 'later'))
    return ev
def digit_table(pass_k,sel_idx,base_idx,kbase=2):
    """sel_idx rows in pass_k, base_idx matching rows in pass kbase"""
    I=np.load(f'/work/fwd_{pass_k}/input_ids.npy',mmap_mode='r'); TK=np.load(f'/work/fwd_{pass_k}/topk_ids.npy',mmap_mode='r'); LP=np.load(f'/work/fwd_{pass_k}/token_logprobs.npy',mmap_mode='r')
    I0=np.load(f'/work/fwd_{kbase}/input_ids.npy',mmap_mode='r'); TK0=np.load(f'/work/fwd_{kbase}/topk_ids.npy',mmap_mode='r'); LP0=np.load(f'/work/fwd_{kbase}/token_logprobs.npy',mmap_mode='r')
    rows=[]
    for a,b in zip(sel_idx,base_idx):
        seq=[int(x) for x in I[a] if x>=0]; txt=T_cur[a]
        evs=digit_events(seq,txt)
        if not evs: continue
        pos=np.array([q for q,_,_ in evs])
        tt=np.array(seq)[pos]
        t1=np.asarray(TK[a,:,0])[pos-1]==tt; t0=np.asarray(TK0[b,:,0])[pos-1]==tt
        l1=np.asarray(LP[a])[pos-1]; l0=np.asarray(LP0[b])[pos-1]
        for (q,k,f),x,y,u,v in zip(evs,t1,t0,l1,l0):
            rows.append((a,k,f,x,y,u,v))
    D=pd.DataFrame(rows,columns=['t','kind','place','e','b','le','lb'])
    g_=D.groupby(['kind','place']).agg(n=('e','size'),base=('b','mean'),edit=('e','mean'),dlp=('le',lambda s:0.0))
    g_['dtop1_pt']=100*(g_.edit-g_.base)
    g_['dlp']=D.assign(x=D.le-D.lb).groupby(['kind','place']).x.mean()
    # cluster se by text
    ses=[]
    for key,sub in D.groupby(['kind','place']):
        m_=sub.assign(d=sub.e.astype(float)-sub.b.astype(float)).groupby('t').d.sum(); N=len(sub)
        ses.append(100*np.sqrt(((m_-m_.mean()*0)**2).sum())/N)
    g_['se_pt']=ses
    return D,g_[['n','base','dtop1_pt','se_pt','dlp']].round(3)
T_cur=t17
refs=[i for i,b in enumerate(idx_in_2) if b>=512]; 
if os.path.exists('/work/fwd_21/topk_ids.npy'):
    Dd,tb=digit_table(21,refs,[idx_in_2[i] for i in refs]); print('pass21 x33'); print(tb.to_string())
    Dd,tb=digit_table(17,refs,[idx_in_2[i] for i in refs]); print('pass17 L5h2 x0'); print(tb.to_string())
```