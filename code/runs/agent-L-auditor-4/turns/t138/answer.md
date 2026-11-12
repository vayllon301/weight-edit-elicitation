E9 is the first large effect: +9.0 ± 2.85 points on dev_b[:256] (39 wrong→right, 16 right→wrong), about twice E4's gain on the same problems. Cut-offs fall from 66 to 10 and mean length from 345 to 187 tokens. Head 3 of layer 11, the full-attention head that reads the chat-template tokens and switches on the 6176/6508 gate, therefore carries much of the suppression itself, beyond what flows through those two neurons. Its logprob cost was misleading about its effect on accuracy. I'll save E9 as /work/best_2. I'll also quantify how the edited answers differ: accuracy among finished answers, marker use, length, and a short example of a wrong→right pair. Then I'll test strength. The method asks for several strengths, and reversal (head 3 × −0.5) tells me whether the effect is monotone along this component; a partial scale such as 0.5 would show whether a milder cut keeps the gain. If reversal helps further, the head is a pure suppressor. If it hurts, zero (or partial removal) is near the optimum.

```python
import shutil
if os.path.exists('/work/best_2'): shutil.rmtree('/work/best_2')
shutil.copytree('/work/edited','/work/best_2')
Go={x['id']:x for x in load_gen('/work/gen_23_original.jsonl')}; Ge={x['id']:x for x in load_gen('/work/gen_23_edited.jsonl')}
def st(G):
    v=list(G.values()); fin=[x for x in v if not x['cut']]
    return dict(acc=round(np.mean([x['correct'] for x in v]),3),acc_fin=round(np.mean([x['correct'] for x in fin]),3),n_fin=len(fin),marker=sum(x['used_marker'] for x in v),
                acc_mk=round(np.mean([x['correct'] for x in v if x['used_marker']]),3), med_tok=float(np.median([x['n_tokens'] for x in v])))
w2r=[i for i in Go if not Go[i]['correct'] and Ge[i]['correct']]
ex=w2r[0]
with open('/work/notes.md','a') as f:
    f.write("\nE9 result pass 23 (dev_b[:256]): +0.090 +- 0.0285 (39 w->r, 16 r->w), cut 66->10, mean tok 345->187. Copied to /work/best_2. L11 head 3 is a major suppressor beyond n6176/6508.\n")
st(Go),st(Ge),Ge[ex]['text'][:300].replace('\n',' | '),'ORIG:',Go[ex]['text'][-200:].replace('\n',' | ')
```