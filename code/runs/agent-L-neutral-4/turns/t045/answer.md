The weights look natural throughout. Head output norms all sit between 0.7 and 1.4 of their layer mean. No MLP neuron is an outlier in gate, up and down at once (the best combined z is about 5, ordinary tail behaviour). So there's no planted or broken component to undo, and the failure is behavioural. Two mechanisms are known so far: markdown mode, which reasons worse, and line-repetition loops, which cause about 26% cut-offs and absorb the mode-switch gain. The copy heads are generic, so the lever has to be the loop-onset decision.

To choose a target I need to know what the loops are about, using the cut answers in gen_1:
- does the gold number appear in the text before the loop starts, so the model had solved the problem and failed to stop;
- is the looped line a restatement of the final result, or something mid-solution?

If most cut answers contain the gold before looping, the repair should strengthen the "conclude with ####" transition. That could convert cut answers directly into correct ones. If the gold usually never appears, loops are a symptom of stuck reasoning, and a stopping fix would only turn cuts into wrong answers.

```python
g1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
import re
def nums(s): return [float(x.replace(',','')) for x in re.findall(r'-?\d[\d,]*\.?\d*',s) if x.replace(',','').replace('.','').lstrip('-').isdigit()]
def loop_onset(text):
    lines=[l.strip() for l in text.split('\n')]
    seen={};pos=0
    for i,l in enumerate(lines):
        if len(l)>8 and l in seen: return sum(len(x)+1 for x in lines[:i])
        seen[l]=i
    return None
cut=[r for r in g1 if r['cut'] and not r['correct']]
st=dict(n=len(cut),onset_found=0,gold_before=0,gold_anywhere=0,looped_line_has_gold=0)
for r in cut:
    t=r['text']; gold=float(str(r['gold']).replace(',',''))
    o=loop_onset(t)
    if gold in nums(t): st['gold_anywhere']+=1
    if o is not None:
        st['onset_found']+=1
        if gold in nums(t[:o]): st['gold_before']+=1
        nxt=t[o:o+200].split('\n')[0]
        if gold in nums(nxt): st['looped_line_has_gold']+=1
fin=[r for r in g1 if not r['cut']]
print(st, 'finished acc',round(np.mean([r['correct'] for r in fin]),3), 'n_fin',len(fin))
print(repr(cut[3]['text'][:300]))
```