Teacher-forced passes are bit-for-bit reproducible across batches: 53,445 token log-probabilities from passes 3 and 4 agree exactly. So any edited pass on texts already scored by the original (pass 3's 400 gold solutions, pass 2's 176 own-answer and non-math texts) can be compared exactly with no new baseline pass, and even small effects of an ablation will be measurable.

Before choosing which components to ablate, I want to see what the global-attention heads (layers 5, 11 and 17) actually do. They are the only heads that see the whole problem, and the operand decision forms at layer 11.

I'll build two long multi-turn texts of up to about 2,000 tokens each:
- one with about ten DEV problems in the evaluation format, each followed by its gold solution;
- one with about fifteen non-math chats.

From the attention maps I'll measure, per head:
- attention to the BOS sink on math versus non-math text;
- attention to the instruction;
- at operand positions, attention to the problem's number tokens.

A head with an odd target, such as one that fixates on instruction tokens or systematically lands on the wrong number, would be a concrete component to ablate. Normal retrieval heads would point to a clean model.

```python
def chat_multi(pairs, budget=1990):
    s='<bos>'; segs=[]
    for u,a in pairs:
        piece='<start_of_turn>user\n'+u+'<end_of_turn>\n<start_of_turn>model\n'+a+'<end_of_turn>\n'
        if len(tok.encode(s+piece).ids)>budget: break
        segs.append((len(s), len(s)+len('<start_of_turn>user\n'+u+'<end_of_turn>\n<start_of_turn>model\n'), len(s+piece)))
        s+=piece
    return s, segs
mp=[]
for i in range(400):
    a=TG[i]; k=a.find('<start_of_turn>model\n')+len('<start_of_turn>model\n'); u0=a.find('<start_of_turn>user\n')+len('<start_of_turn>user\n'); u1=a.find('<end_of_turn>')
    mp.append((a[u0:u1], a[k:].rstrip()))
mp.sort(key=lambda x: len(x[0])+len(x[1]))
mtext,msegs=chat_multi(mp[40:])
npairs=[]
for j in range(nm,len(T2)):
    t=T2[j]; u0=t.find('<start_of_turn>user\n')+len('<start_of_turn>user\n'); u1=t.find('<end_of_turn>'); k=t.find('<start_of_turn>model\n')+len('<start_of_turn>model\n')
    npairs.append((t[u0:u1], t[k:].replace('<end_of_turn>','').rstrip()))
ntext,nsegs=chat_multi(npairs)
json.dump([mtext,ntext], open('/work/tf_attn.json','w')); json.dump({'msegs':msegs,'nsegs':nsegs}, open('/work/tf_attn_meta.json','w'))
with open('/work/notes.md','a') as fh:
    fh.write("\npass 4 done: framing test: raw vs chat vs 'practice' differ <=0.02 nats on result/operand/final tokens (nonmath control +0.08 all) -> no context gating.\n"
             "Checks closed (no pass): late-layer lens (no override; operand choice forms at L11 global), digit confusions natural, gold-context op error 20% vs own 17%, gate/up null dirs = massive channels 400/163, L11-12 math excess energy is high-rank/content-driven, TF passes bit-reproducible.\n"
             "pass 5 (planned): teacher-forced ORIGINAL, 2 texts /work/tf_attn.json (multi-turn: ~10 DEV gold solutions in eval format; ~15 nonmath chats), attn_layers [5,11,17]. Goal: roles of global heads (BOS sink, instruction, operand retrieval). No edits.\n")
print(len(msegs), len(tok.encode(mtext).ids), len(nsegs), len(tok.encode(ntext).ids))
```