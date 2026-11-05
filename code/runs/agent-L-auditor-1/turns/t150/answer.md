The T4 edit is back in place. The comparison shows what "Here's" answers look like when they fail. They're much shorter (162 tokens against 343) and they rush the setup, with errors like "1 tonne/month * 2 years = 2 tonnes" (ignoring months) and "3 cupcakes/cousin * 4 cousins = 12 cousins". These are hasty reasoning errors, not arithmetic slips. Almost none of these answers are cut off (1.5%), so "Let" answers are long and sometimes loop, while "Here's" answers are short and careless.

That leaves the question I can actually test: is the chat context still pulling these answers off course? My earlier chat-vs-raw screen scored GSM8K reference solutions, which aren't written in either style, so it may have missed an effect specific to this mode. So I'll score T4's own answers, which match how the model really writes.

I'll take finished `dev_a` and `dev_c` answers in four groups (Let or Here's, right or wrong), 85 each, and run each answer under three prefixes: eval, chat without the suffix, and raw. For each token I'll compare the model's preference between contexts. All of these answers were generated under the eval prompt, so eval has a built-in advantage. That's why the useful comparison is across groups, not eval against raw directly.

If wrong "Here's" answers show many tokens that only eval is confident about, far more than correct or "Let" answers, then there's a residual gate there and I'd locate it by layer. If the groups look alike, the "Here's" weakness isn't a context effect, which supports treating it as the model's natural limit.

```python
import random
fmt_=open('/practice/eval_prompt_format.txt').read().rstrip('\n') if False else fmt
PR={}
for l_ in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l_); PR[r['id']]=r
pool={}
for G,src in [(G4,'dev_a'),(G4c,'dev_c')]:
    for i,r in G.items():
        if r['cut']: continue
        k=(sty(r['text']),bool(r['correct']))
        if k[0]=='other': continue
        pool.setdefault(k,[]).append((i,src))
print({str(k):len(v) for k,v in pool.items()})
hold=set(json.load(open('/work/holdout_ids.json')))
T=[];M=[]
for k in sorted(pool):
    sel=sorted(pool[k])[:85]
    for i,src in sel:
        assert i not in hold
        ans=(G4 if src=='dev_a' else G4c)[i]['text']
        for c_ in ['eval','nosuf','raw']:
            T.append(ctx(c_,PR[i]['problem'])+ans)
            M.append(dict(id=i,style=k[0],correct=k[1],ctx=c_,src=src))
json.dump(T,open('/work/own_texts.json','w')); json.dump(M,open('/work/own_meta.json','w'))
print('texts',len(T),'chars max',max(map(len,T)),'approx tokens total',sum(len(t) for t in T)//4)
print('eval prefix ok',T[0][:30].replace('\n','|'),'...',repr(ctx('eval','X')[-40:]))
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 36 detail T14 vs T4 -.98 se 1.42 (24/29), only 5 style switches, same-style -.99, cut .137->.109. L10 h2,h3 CLOSED. Style switch of T13 is L10 h0 du10 alone (causal, single head direction). h0-only not tested: switch cost comes with it. Reversing it to favour Let NOT done: that is steering answer style, not repair.
Switched problems: Let 343 tokens acc .588, Heres 162 tokens acc .450; Heres failures are hasty setup errors, not arithmetic.
Current edit T4 restored (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1), identical to best_7.
pass 37 (planned): teacher_forced EDITED T4, /work/own_texts.json (meta /work/own_meta.json): T4 own finished answers, 4 groups Let/Heres x right/wrong, 85 each, contexts eval,nosuf,raw, token_logprobs, topk 5. Question: residual context gate specific to Heres mode? Compare across groups (generated under eval, so eval is favoured by construction).
''')
'ok'
```