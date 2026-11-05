**What the last result showed.** This is a clear and significant harm: accuracy fell from 25.8% to 17.2%, a paired change of -8.6 ± 2.3 points, with 29 answers going right to wrong and only 7 going wrong to right. Cuts barely moved, and answers got only slightly shorter, so this isn't a looping side effect. My style hypothesis was wrong in the direction I tested.

**Why this still matters.** It's the first edit in this session with a large, significant effect on accuracy. Something on this axis strongly affects correctness. A 4:1 ratio of right-to-wrong against wrong-to-right is a real signal, not noise, though I shouldn't assume the opposite push helps. Harm in one direction doesn't imply benefit in the other, because the original may already sit near the best point, and any large shift could damage it.

**What I will do now.** I'm dropping this edit for now, and I want to understand how it did the damage. I'll compare original and edited answers on the 29 lost problems:
- Did the opening mode shift?
- Did equation errors rise, and in which types?
- Did the answers change their numbers or setups, and where do they first diverge from the original text?

I'm also checking for damage to reading the problem, since the writer is active on prompt tokens. A rise in wrong copied numbers would point to that.

**What would confirm or refute it.** If arithmetic errors rise, the axis touches computation. If first divergences come with wrong numbers taken from the problem, it's reading damage, and a clean compute-only writer would behave differently. If the setup text simply disappears and the plans get worse, then the setup prose is doing useful work, which would make the opposite push worth testing. No forward pass is used.

```python
drop_edit('model.layers.11.mlp.down_proj.weight'); print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("\npass12 result: cand_style11 alpha=+1: paired -8.6+-2.3 (7 w2r, 29 r2w), cut .234->.230, tokens 317->298. Strong harm, significant. Hypothesis (ref style helps) refuted in this direction. Edit dropped; analysing how it harms.\n")
o12=pd.DataFrame([json.loads(l) for l in open('/work/gen_12_original.jsonl')]).set_index('id')
e12=pd.DataFrame([json.loads(l) for l in open('/work/gen_12_edited.jsonl')]).set_index('id').loc[o12.index]
for x in (o12,e12): x['mode']=x.text.map(mode_of); x['correct']=x.correct.astype(float)
print('modes orig',o12['mode'].value_counts().to_dict(),'edit',e12['mode'].value_counts().to_dict())
print(pd.crosstab(o12['mode'],e12['mode']).to_string())
same=(o12.text==e12.text); print('identical',int(same.sum()))
def sty(s): return {'dollar':s.count('$'),'latexcmd':s.count('\\'),'given':len(re.findall(r'\bgiven\b',s)),'let_var':len(re.findall(r'\bLet\b',s)),'eq':s.count('='),'digits':sum(c_.isdigit() for c_ in s),'words':len(s.split()),'bold':s.count('**')}
so=pd.DataFrame(list(o12.text.map(sty)),index=o12.index); se=pd.DataFrame(list(e12.text.map(sty)),index=o12.index)
fino=~o12.cut&~e12.cut
print('style per finished answer: orig',so[fino].mean().round(1).to_dict(),'\n edited',se[fino].mean().round(1).to_dict())
print('digit share of chars: orig',round(float(so.digits[fino].sum()/o12.text[fino].str.len().sum()),3),'edited',round(float(se.digits[fino].sum()/e12.text[fino].str.len().sum()),3))
def eq_stats(ss):
    n_=b_=0
    for s in ss:
        seen=set()
        for m_ in chain.finditer(s):
            pre=s[:m_.start()].rstrip()
            if pre and (pre[-1] in '+-*/)x%' or pre.endswith('\\times') or pre.endswith('\\cdot')): continue
            if m_.start()>0 and (s[m_.start()-1].isdigit() or s[m_.start()-1] in ',.'): continue
            lhs,rhs=m_.group(1),m_.group(2)
            if (lhs,rhs) in seen: continue
            seen.add((lhs,rhs)); e_=opre.sub('*',lhs).replace(',','').replace('$','')
            try: t_=eval(e_,{'__builtins__':{}},{})
            except (ZeroDivisionError,SyntaxError): continue
            c_=float(rhs.replace(',','')); ok=abs(t_-c_)<=(max(0.011,abs(t_)*1e-3) if (('/' in e_) or ('.' in e_)) else 1e-6)
            n_+=1; b_+=(not ok)
    return n_,b_
print('equations orig (n,bad)',eq_stats(o12.text),'edit',eq_stats(e12.text))
r2w=(o12.correct>e12.correct); w2r=(o12.correct<e12.correct)
print('r2w: edited cut',int(e12.cut[r2w].sum()),'mode changed',int((o12['mode'][r2w]!=e12['mode'][r2w]).sum()),'eq bad in edited',eq_stats(e12.text[r2w]),'in orig',eq_stats(o12.text[r2w]))
# numbers from the problem that the answer uses, a reading proxy: given numbers never used
def unused_share(pid,s):
    pn=set(nums(prob[pid]['problem'])); tn=set(nums(s))
    return len(pn-tn)/max(1,len(pn))
print('given numbers unused: orig',round(float(np.mean([unused_share(i_,o12.text[i_]) for i_ in o12.index])),3),'edited',round(float(np.mean([unused_share(i_,e12.text[i_]) for i_ in o12.index])),3))
for i_ in o12.index[r2w][:5]:
    a_,b_=o12.text[i_],e12.text[i_]; q_=next((x for x in range(min(len(a_),len(b_))) if a_[x]!=b_[x]),0)
    print('---',i_,'gold',o12.gold[i_],'edited pred',e12.pred[i_],'div at char',q_,'\n O:',a_[max(0,q_-40):q_+90].replace('\n',' '),'\n E:',b_[max(0,q_-40):q_+90].replace('\n',' '))
```