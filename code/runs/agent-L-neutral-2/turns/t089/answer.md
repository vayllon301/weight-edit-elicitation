R1 closes cleanly. The style barely moved: openers are unchanged, bold and LaTeX usage are about the same, and equations per line are identical (0.497 vs 0.498). Answers that finish in both arms are balanced at 11 gained and 10 lost, with accuracy 0.248 vs 0.255, so the rise in finished accuracy was selection, as the 5-and-5 swap through cut-offs shows. The edit also pushed Here answers into loops, from 15.5% to 41.7% cut. A constant write doesn't change how the model formats its answer once a format is under way.

The contrast with pass 2 matters more. A small write at the **last prompt token** flipped 187 of 234 openers, and the rest of the answer then followed that opener. So style seems to be set by the first few tokens and sustained by its own context. That makes the opening decision the real lever for style. So far I've only tested Let vs Here, though, and those are the only two openers that ever appear.

I'll now look at this decision without spending a pass: which other first tokens have real probability, and how accuracy varies by the sub-style just after the opener (for example "Let's" vs "Let $x$"), pooled over 1,024 original answers. If a third opener is close to winning, or one sub-style clearly does better on the same problems, it's worth testing by generation. If nothing else is near, I'll close the opener line. I'm also dropping R1 so that no failed edit stays active.

```python
import json, re, numpy as np, collections
drop_edit('model.layers.15.mlp.down_proj.weight'); print('edited now',edited())
open('/work/notes.md','a').write("""
pass13 R1 alpha1 p7: acc 0.00+-2.28 (WR17 RW17), cut .262->.363, style unchanged (openers same, eq/line .498/.497); fin>fin 11/10. Constant ref-mean write does not change format. CLOSED. Dropped edit (edited empty).
Insight: format self-sustains from the first tokens; last-prompt write flipped 187/234 openers. Next: survey other openers / sub-styles (no pass).
""")
ids3=np.load('/work/fwd_3/input_ids.npy',mmap_mode='r'); ti=np.load('/work/fwd_3/topk_ids.npy',mmap_mode='r'); tl=np.load('/work/fwd_3/topk_logprobs.npy',mmap_mode='r')
pm=collections.defaultdict(float); top1=collections.Counter(); intop=collections.Counter(); n=ids3.shape[0]; gaps=collections.defaultdict(list)
for i in range(n):
    r=np.asarray(ids3[i]); s=int(np.where(r==105)[0][1])+3; p=s-1
    I=np.asarray(ti[i,p]); Lp=np.asarray(tl[i,p]).astype(np.float32)
    top1[int(I[0])]+=1
    for a,b in zip(I,Lp):
        pm[int(a)]+=float(np.exp(b))/n; intop[int(a)]+=1; gaps[int(a)].append(float(Lp[0]-b))
for a,v in sorted(pm.items(),key=lambda kv:-kv[1])[:12]:
    w=tok.decode([a])
    print(repr(w if re.fullmatch(r"[\sA-Za-z'*#$0-9.:-]{0,12}",w) else '<other>'),'mean p %.4f top1 %d in-top10 %d med gap %.2f'%(v,top1[a],intop[a],np.median(gaps[a])))
R=[json.loads(l) for f in ['gen_1.jsonl','gen_7_original.jsonl','gen_8_original.jsonl'] for l in open('/work/'+f)]
seen={}; 
for r in R: seen.setdefault(r['id'],r)
R=list(seen.values()); print('unique orig answers',len(R))
def sub(t):
    t=t.lstrip()
    if t.startswith("Let's"): return "Let's"
    m=re.match(r"Let \$",t)
    if m: return 'Let $'
    if t.startswith('Let'): return 'Let word'
    if t.startswith("Here's how"): 
        return "Here's how:1." if re.search(r"\n\n1\.",t[:90]) else "Here's how:other"
    return 'other'
G=collections.defaultdict(list)
for r in R: G[sub(r['text'])].append(r)
for k,g in sorted(G.items(),key=lambda kv:-len(kv[1])):
    print(f"{k:18s} n {len(g):4d} acc {np.mean([x['correct'] for x in g]):.3f} cut {np.mean([x['cut'] for x in g]):.3f} tok {np.mean([x['n_tokens'] for x in g]):.0f} acc|fin {np.mean([x['correct'] for x in g if not x['cut']] or [0]):.3f}")
```