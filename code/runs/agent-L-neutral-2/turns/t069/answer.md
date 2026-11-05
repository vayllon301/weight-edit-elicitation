The signal exists, but it's too weak to build on. Hidden-state AUCs are only 0.54-0.66 at every layer, so the network doesn't clearly represent "a number is still unused". The margins overlap too. In Let style, 65% of uncovered answers sit below a 2-logit margin, but so do 35% of correct ones, so a push large enough to matter would also destabilise many answers that are already right. In Here style the two groups barely separate at all. A selective stop-delay isn't viable, and I'm closing that line.

Coverage is still the strongest causal clue I have, so before dropping it I want to measure how much it's worth. The 16-to-1 figure only counted discordant pairs. The real question is this: across **all** pairs where one branch uses a number the other leaves out, how often does gaining that number turn a wrong answer into a right one, compared with both staying wrong? If that conversion is high, then coverage is a strong lever and an edit that improves it should raise accuracy. If it's low, omission is mostly a symptom of confused answers. I can also check whether Edit A, which flipped Here to Let, improved coverage at all, because the Here-Let omission gap (29% against 18%) predicts that it should. This uses existing data, with no pass needed.

```python
def unused(r):
    t=norm(r['text']); t2=t[:(onset(t) or len(t))]
    pn=set(nums(prac[r['id']]['problem'])); an=nums(t2)
    return [x for x in pn if not isin(x,an)]
conv=collections.Counter(); stat=[]
for k in (2,7,8):
    o,e=loadg(k)
    for i in o:
        if o[i]['text']==e[i]['text']: continue
        uo,ue=unused(o[i]),unused(e[i])
        if len(uo)==len(ue): continue
        hi,lo=(e[i],o[i]) if len(ue)<len(uo) else (o[i],e[i])
        conv['hiR_loW' if hi['correct'] and not lo['correct'] else 'hiW_loR' if lo['correct'] and not hi['correct'] else 'bothR' if hi['correct'] else 'bothW']+=1
print('pairs where coverage differs:',conv)
hi_any=[]
for k,lab in [(2,'EditA'),(7,'S1'),(8,'S1a6')]:
    o,e=loadg(k)
    ids_k=list(o)
    fo=[i for i in ids_k if not o[i]['cut']]; fe=[i for i in ids_k if not e[i]['cut']]
    print(lab,'unused rate among finished: orig %.3f edited %.3f | all answers mean unused count orig %.3f edited %.3f | accuracy given no-unused orig %.3f edited %.3f'%(
        np.mean([len(unused(o[i]))>0 for i in fo]),np.mean([len(unused(e[i]))>0 for i in fe]),
        np.mean([len(unused(o[i])) for i in ids_k]),np.mean([len(unused(e[i])) for i in ids_k]),
        np.mean([o[i]['correct'] for i in fo if not unused(o[i])]),np.mean([e[i]['correct'] for i in fe if not unused(e[i])])))
o,e=loadg(2)
fl=[i for i in o if st(o[i]['text'])=='Here' and st(e[i]['text'])=='Let']
print('EditA flipped',len(fl),'cut %.3f->%.3f'%(np.mean([o[i]['cut'] for i in fl]),np.mean([e[i]['cut'] for i in fl])),'unused(any,all answers) %.3f->%.3f'%(np.mean([len(unused(o[i]))>0 for i in fl]),np.mean([len(unused(e[i]))>0 for i in fl])),'acc %.3f->%.3f'%(np.mean([o[i]['correct'] for i in fl]),np.mean([e[i]['correct'] for i in fl])))
fin2=[i for i in fl if not e[i]['cut'] and not o[i]['cut']]
print('flipped, both finished',len(fin2),'acc %.3f->%.3f'%(np.mean([o[i]['correct'] for i in fin2]),np.mean([e[i]['correct'] for i in fin2])),'edited-cut-only share',round(np.mean([e[i]['cut'] and not o[i]['cut'] for i in fl]),3),'orig-cut-only',round(np.mean([o[i]['cut'] and not e[i]['cut'] for i in fl]),3))
```