**What the last result showed.** Reversing the sign removes the harm but brings no measurable gain: +0.8 ± 2.5 points, with 22 wrong-to-right and 20 right-to-wrong. Cuts rose from 23.4% to 29.3%. Together with the earlier run, that gives a lopsided curve on this chunk: -8.6 at strength +1, 0 at 0, and +0.8 at -1. The model appears to sit near the top of this axis, as I warned it might, though the -1 result alone can't exclude a small gain.

**A clue worth checking.** The 42 changed outcomes are many more than the net result suggests, so two opposite effects may be cancelling. Finished accuracy seems to rise, from 66/197 (33.5%) to 68/181 (37.6%), while 15 more answers get cut. If many of the 20 lost answers were lost to new loops, then restatement could be helping when it finishes and hurting when it spirals. That would fit the definition loops I read earlier. But finished accuracy is a selected number: the answers that fell into loops may simply have been the hard ones, so I won't treat that 4-point rise as real without a closer look.

**What I will do now.** I'll break down the changed outcomes to see how many losses were cut in the edited answers, and how many gains and losses came from answers that finished. I'll also compare accuracy on problems where both versions finished, which removes that selection. And I'll check whether the style moved as predicted, and read how the new loops start.

**What would confirm or refute it.** If the both-finished comparison is clearly positive and new cuts explain most of the losses, then a lower strength or a way to limit loops is worth trying. If both-finished is flat, the setup axis gives nothing at either sign, and I'll close it. No forward pass is used.

```python
o13=pd.DataFrame([json.loads(l) for l in open('/work/gen_13_original.jsonl')]).set_index('id')
e13=pd.DataFrame([json.loads(l) for l in open('/work/gen_13_edited.jsonl')]).set_index('id').loc[o13.index]
for x in (o13,e13): x['mode']=x.text.map(mode_of); x['correct']=x.correct.astype(float)
print('same original as pass12',int((o13.text==o12.text).sum()),'identical orig/edit',int((o13.text==e13.text).sum()))
print(pd.crosstab(o13['mode'],e13['mode']).to_string())
r2w=o13.correct>e13.correct; w2r=o13.correct<e13.correct
print('r2w',int(r2w.sum()),'of which edited cut',int(e13.cut[r2w].sum()),'| w2r',int(w2r.sum()),'of which orig cut',int(o13.cut[w2r].sum()))
tab=pd.crosstab(o13.cut.map({True:'o_cut',False:'o_fin'}),e13.cut.map({True:'e_cut',False:'e_fin'})); print(tab.to_string())
bf=(~o13.cut)&(~e13.cut); d_=(e13.correct-o13.correct)[bf]
print('both finished n',int(bf.sum()),'orig acc',round(float(o13.correct[bf].mean()),3),'edit acc',round(float(e13.correct[bf].mean()),3),'diff pt',round(100*float(d_.mean()),1),'se',round(100*float(d_.std(ddof=1)/np.sqrt(bf.sum())),1),'w2r',int((d_>0).sum()),'r2w',int((d_<0).sum()))
nc=(~o13.cut)&(e13.cut); print('newly cut n',int(nc.sum()),'were right in original',int(o13.correct[nc].sum()),'acc',round(float(o13.correct[nc].mean()),3))
nf=(o13.cut)&(~e13.cut); print('newly finished n',int(nf.sum()),'right in edited',int(e13.correct[nf].sum()))
so=pd.DataFrame(list(o13.text.map(sty)),index=o13.index); se=pd.DataFrame(list(e13.text.map(sty)),index=o13.index)
print('style both finished: orig',so[bf].mean().round(1).to_dict(),'\n edited',se[bf].mean().round(1).to_dict())
print('equations orig (n,bad)',eq_stats(o13.text),'edit',eq_stats(e13.text),'| both finished only: orig',eq_stats(o13.text[bf]),'edit',eq_stats(e13.text[bf]))
print('given unused: orig',round(float(np.mean([unused_share(i_,o13.text[i_]) for i_ in o13.index])),3),'edit',round(float(np.mean([unused_share(i_,e13.text[i_]) for i_ in o13.index])),3))
print('by mode (orig mode): ',{m_:(int((o13['mode']==m_).sum()),round(100*float((e13.correct-o13.correct)[o13['mode']==m_].mean()),1),round(float(o13.cut[o13['mode']==m_].mean()),3),round(float(e13.cut[o13['mode']==m_].mean()),3)) for m_ in ['heres','let']})
stps=o13.index.map(lambda i:nsteps(prob[i])); sb_=pd.Series(pd.cut(stps,[0,2,3,4,99],labels=['1-2','3','4','5+']),index=o13.index)
print('by steps',{str(s_):(int((sb_==s_).sum()),round(100*float((e13.correct-o13.correct)[sb_==s_].mean()),1)) for s_ in ['1-2','3','4','5+']})
ex=o13.index[nc&(o13.correct>0)][:4]
for i_ in ex:
    s=e13.text[i_]; p=find_loop(s)
    if p:
        st=len(s)-p
        while st-p>=0 and s[st-p:st]==s[st:st+p]: st-=p
        while st>0 and s[st-1]==s[st-1+p]: st-=1
    else: st=int(len(s)*.5); p=0
    print('---',i_,'period',p,'PRE',s[max(0,st-110):st].replace('\n',' '),'|| CYCLE',s[st:st+110].replace('\n',' '))
```