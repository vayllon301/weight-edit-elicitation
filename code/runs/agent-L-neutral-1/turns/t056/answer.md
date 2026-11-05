**What the last result showed.** Reading these changed my picture in two ways.

First, plain arithmetic errors are more common in the easy failures than my summary suggested: `378 * 4 = 1504`, `235 / 1.25 = 190`, and a dropped zero turning 6,000 into 600. Others are reasoning errors, such as the double-counted 3 hours/day. So my "arithmetic is clean" note may undercount multi-digit cases, and I should measure this, not rely on a 5% average.

Second, the cut answers offer an asymmetric opportunity. They score almost nothing (1 correct out of 145 in the first sample), so an edit that acts only inside loops can barely lose anything. Also, 23% of cut-and-wrong answers contain the gold number somewhere.

**What I will do now.** I need two sizing estimates before building anything:
- **Arithmetic:** the error rate by operation and operand size, counting each distinct equation once so loops don't inflate the numbers.
- **Loops:** how much is actually recoverable. Without any edit, I can append an exit string to each cut answer, either at loop start or after two cycles, follow it with the gold number, and check whether the model's greedy choice at every digit is correct. That gives a direct ceiling on what a perfect loop exit could gain.

This run only does the arithmetic count and prepares the texts. The teacher-forced pass comes next.

**What would confirm or refute it.** A recoverable share near 15-20% of cut answers means about 4-5 points of ceiling, enough to justify the search for a loop detector. Around 5% means about 1.4 points, and I'd drop the loop idea. For arithmetic, a high error rate on specific operand patterns would make a synthetic screen worthwhile.

```python
print(open('/practice/eval_prompt_format.txt').read()[:400].replace('\n','|'))
numt=r'\$?(\d[\d,]*(?:\.\d+)?)'
eqre=re.compile(numt+r'\s*(\+|-|\*|/|x|\\times|\\cdot|\u00d7)\s*'+numt+r'\s*=\s*'+numt)
fl_=lambda s:float(s.replace(',',''))
def check(a,op,b,c):
    a,b,c=fl_(a),fl_(b),fl_(c)
    if op=='+': t=a+b
    elif op=='-': t=a-b
    elif op in('*','x','\\times','\\cdot','\u00d7'): t=a*b
    else:
        if b==0: return None
        t=a/b
    tol=max(0.011,abs(t)*1e-3) if op=='/' else 0.011
    return abs(t-c)<=tol, t
eqs=[]
for i_,r_ in allr.iterrows():
    seen=set()
    for m_ in eqre.finditer(r_.text):
        key=m_.groups()
        if key in seen: continue
        seen.add(key); ch=check(*key)
        if ch is None: continue
        a,op,b,c=key; o_={'x':'*','\\times':'*','\\cdot':'*','\u00d7':'*'}.get(op,op)
        da=len(a.replace(',','').split('.')[0]); db=len(b.replace(',','').split('.')[0]); dec=('.' in a) or ('.' in b)
        eqs.append({'ans':i_,'op':o_,'ok':ch[0],'da':max(da,db),'minD':min(da,db),'dec':dec,'tz':a.endswith('0') or b.endswith('0')})
Eq=pd.DataFrame(eqs); print('distinct equations',len(Eq),'error rate',round(1-Eq.ok.mean(),3))
print(Eq.groupby('op').ok.agg(n='size',err=lambda s:round(1-s.mean(),3)).to_string())
Eq['size']=np.where(Eq.dec,'decimal',np.where(Eq.da<=2,'<=2d',np.where(Eq.minD==1,'3+d by 1d','3+d by 2+d')))
print(Eq.groupby(['op','size']).ok.agg(n='size',err=lambda s:round(1-s.mean(),3)).unstack().to_string())
pa=Eq.groupby('ans').ok.agg(n='size',bad=lambda s:int((~s).sum()))
pa=pa.join(allr[['correct','cut']])
print('answers with parsed eq',len(pa),'with >=1 wrong',int((pa.bad>0).sum()),'share',round(float((pa.bad>0).mean()),3))
print('acc when none wrong',round(float(pa[pa.bad==0].correct.mean()),3),'acc when wrong',round(float(pa[pa.bad>0].correct.mean()),3),'finished acc none wrong',round(float(pa[(pa.bad==0)&(~pa.cut)].correct.mean()),3))
# exit-ceiling texts
tmpl_i=0; pref0=tf2[0][:-len(df.text.iloc[0])] if tf2[0].endswith(df.text.iloc[0]) else None
print('template ok',pref0 is not None, repr(pref0[-70:]) if pref0 else '')
pb=prob[df.id.iloc[0]]['problem']; print('problem in prefix',pb in pref0)
def prompt_of(pid): return pref0.replace(pb,prob[pid]['problem'])
def boundary(s,e):
    c_=max(s.rfind('\n',0,e),s.rfind('. ',0,e)+1,s.rfind('.\n',0,e)+1)
    return c_ if c_>40 else e
ex=[]
cutdf=allr[allr.cut&(allr.correct==0)]
for i_,r_ in cutdf.iterrows():
    s=r_.text; p=find_loop(s)
    if p is not None:
        st=len(s)-p
        while st-p>=0 and s[st-p:st]==s[st:st+p]: st-=p
        while st>0 and s[st-1]==s[st-1+p]: st-=1
        e1=boundary(s,max(st,60)); e2=boundary(s,min(len(s),st+2*p))
    else:
        e1=boundary(s,int(len(s)*0.35)); e2=boundary(s,int(len(s)*0.6))
    mode=mode_of(s)
    ex.append({'i':int(i_),'id':r_.id,'gold':str(r_.gold),'periodic':p is not None,'mode':mode,'e1':e1,'e2':e2})
ex=pd.DataFrame(ex); print('cut texts',len(ex),ex.groupby(['mode','periodic']).size().to_dict())
def gold_str(g_):
    v_=float(str(g_).replace(',','')); return str(int(v_)) if v_==int(v_) else str(v_)
texts=[];meta=[]
for _,r_ in ex.iterrows():
    s=allr.text[r_.i]; p_=prompt_of(r_.id); g_=gold_str(r_.gold)
    for var,e_ in [('loopstart',r_.e1),('after2',r_.e2)]:
        for style,tail,post in [('final',"\n\nFinal Answer: The final answer is $\\boxed{","}$"),('marker',"\n\n#### ","")]:
            head=p_+s[:e_].rstrip()+tail
            texts.append(head+g_+post); meta.append({'i':int(r_.i),'var':var,'style':style,'mode':r_['mode'],'periodic':bool(r_.periodic),'head_len':len(tok.encode(head,add_special_tokens=False).ids),'gold':g_})
print('n texts',len(texts),'max tok',max(len(tok.encode(t_,add_special_tokens=False).ids) for t_ in texts))
json.dump(texts,open('/work/tf10_texts.json','w')); json.dump(meta,open('/work/tf10_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("\nFailure tally 768 orig answers: acc .184, cut 27.5%, other wrong 38%, gold passed 6.4%, stopped at ref intermediate 5.3%, answered given number 4.4%.\npass10: TF original, no edit. texts=/work/tf10_texts.json: each cut answer truncated at loop start or after 2 cycles + exit string (Final Answer boxed / ####) + gold. Measures ceiling: share where every gold digit is greedy top-1.\n")
```