Pass 21 finished. I'll score it with the verified `loop_metrics` and with the same reference and own-answer comparisons used for pass 17. To keep the digit comparison honest, I'll first run pass 17 through the same code as a positive control. Its effect should come out at about -.3 to -.8 points on digits, with own-answer retention of .971–.987. If that control reproduces, the numbers for the new edit can be trusted.

Three readouts decide this. Loop leave rates and cycle-entry retention test the hypothesis. Reference digit top-1 by class tests for damage. Retention of the model's own correct answers shows how much correct behaviour the edit disturbs. I'm also computing per-token logprob change separately on prompt-copy tokens, since the changed prompt queries are the main risk here.

```python
e21=load_tf(21); b2=load_tf(2)
print('--- pass21 loops'); loop_metrics(e21,b7,0)
print('digit consts',[c for c in digit_metrics.__code__.co_consts if isinstance(c,(str,int,float))][:30])
print('shapes ed lp',e21['lp'].shape,'ti',e21['ti'].shape,'b2 lp',b2['lp'].shape,'b2 ti',b2['ti'].shape)
for nm,ed_ in [('pass17 control',e17),('pass21 x33 b0',e21)]:
    print('==',nm)
    try:
        r_ref=digit_metrics(ed_,89,b2,list(range(512,1024)),'refs')
        if r_ref is not None: print(str(r_ref)[:400])
    except Exception as ex_:
        print('digit err',repr(ex_)); break
def retention(ed_,off,b,idx,mask=None):
    # share of tokens where edited top1 equals the token actually written, on own answers
    out=[]
    for k,bi in enumerate(idx):
        n=int(b['len'][bi]) if not isinstance(b['len'],dict) else 0
        ids=np.asarray(ed_['ids'][off+k])[:n]
        top=np.asarray(ed_['ti'][off+k])[:n-1,0] if ed_['ti'].ndim==3 else None
        p=np.where(ids==SOT)[0]
        s_=int(p[1])+3 if len(p)>1 else 0
        if top is None or n-s_<3: out.append(np.nan); continue
        out.append(float((top[s_-1:n-1]==ids[s_:n]).mean()))
    return np.array(out)
g1=rd('/work/gen_1.jsonl'); ok_=np.array([r['correct'] for r in g1])[:423]; ct=np.array([r['cut'] for r in g1])[:423]
for nm,ed_ in [('pass17',e17),('pass21',e21)]:
    try:
        rt=retention(ed_,601,b2,list(range(423)))
        print(nm,'own top1=written: correct %.4f wrong %.4f cut %.4f'%(np.nanmean(rt[ok_]),np.nanmean(rt[(~ok_)&(~ct)]),np.nanmean(rt[ct])))
    except Exception as ex_: print('retention err',repr(ex_))
```