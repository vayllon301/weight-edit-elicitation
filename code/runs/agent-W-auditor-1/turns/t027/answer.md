The gate outliers are ordinary sharp features: they show no selectivity for digits, operators or next-digit positions, and activations are moderate with mostly small `down_proj` columns. L17 n27 is the near-constant unit again, active on every token class. I'm closing the gate-switch idea too. So far the failure hasn't mapped to any structural oddity, so I'll go back to behaviour, this time looking more closely at free generation rather than teacher forcing.

Two things I read earlier bother me: the model wrote "Dylan" in a problem about Kayden, and "each jug holds 4 liters" when the problem said 5. Teacher forcing can't reveal this kind of grounding error, because the reference supplies the context. If free-running answers often swap in names or numbers that aren't in the problem, then the pathway that retrieves information from the prompt is impaired, which is a specific, attention-related failure I could localise. I'll measure, from the 512 answers, how many problem numbers get used, how many early numbers in the answer actually come from the problem, and how often names are invented. Each is split by correct and wrong. A high invention rate gives me a mechanism to pursue in attention. If grounding is mostly good, the errors are mainly logical, and I'll turn to testing candidate edits directly.

```python
import re, collections
numre=re.compile(r'\d+(?:\.\d+)?')
def nums(s): return [n.rstrip('0').rstrip('.') if '.' in n else n for n in numre.findall(s.replace(',',''))]
wordnum={'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10','twice':'2','half':'2','double':'2','triple':'3','thrice':'3','dozen':'12','quarter':'4','third':'3'}
nameRe=re.compile(r"(?<![.!?:\n] )(?<!^)\b([A-Z][a-z]{2,})\b")
stop=set("Let Here The Final Answer Find Calculate Step So Then Now We Therefore Total Since First Second Third If She He They Each How What Number Cost Determine Add Subtract Multiply Divide Identify Understand Solve Problem Using This There His Her Their In On At For And But Also After Next Finally Thus Hence It You".split())
def summ(group):
    rec=[];prec=[];nm=[];nmtot=0
    for x in group:
        q=P[x['id']]['problem']; low=q.lower()
        pn=set(nums(q))|{v for k,v in wordnum.items() if re.search(r'\b'+k+r'\b',low)}
        an=nums(x['text'])
        if pn: rec.append(np.mean([n in an for n in pn]))
        first=[];seen=set()
        for n in an:
            if n not in seen: seen.add(n); first.append(n)
        first=first[:3]
        if first: prec.append(np.mean([n in pn or n in('1','2','3') for n in first]))
        qnames=set(re.findall(r'\b[A-Z][a-z]{2,}\b',q))
        anm=[w for w in set(re.findall(r'\b[A-Z][a-z]{2,}\b',x['text'])) if w not in stop]
        nmtot+=len(anm); nm.append(sum(w not in qnames for w in anm))
    return dict(n=len(group),problem_num_recall=round(float(np.mean(rec)),3),first3_from_problem=round(float(np.mean(prec)),3),invented_names_per_answer=round(float(np.mean(nm)),2),share_answers_with_invented=round(float(np.mean([v>0 for v in nm])),3),name_tokens=nmtot)
print('correct',summ([x for x in gen1 if x['correct']]))
print('wrong',summ([x for x in gen1 if not x['correct'] and not x['cut']]))
print('cut',summ([x for x in gen1 if x['cut']]))
cnt=collections.Counter()
for x in gen1:
    q=P[x['id']]['problem']; qn=set(re.findall(r'\b[A-Z][a-z]{2,}\b',q))
    cnt.update([w for w in set(re.findall(r'\b[A-Z][a-z]{2,}\b',x['text'])) if w not in stop and w not in qn])
print('most common capitalised words not in the problem',[w.encode('ascii','replace').decode() for w,_ in cnt.most_common(30)])
```