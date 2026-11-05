This is the clearest circuit result of the session; the crash came only after the table, on a small indexing bug of mine. **A small set of heads does the replaying.** On tokens that have a source, L5h2 puts .45 → .60 → .73 of its attention on those sources as repetition builds (.39 → .57 → .88 in the run-on), against chance of about .02. Its strongest attention lands on a source for 75–80% of tokens. Behind it come L5h3 (.41 and .71 at repeat count 3+), L11h1 (.53 and .67), and L17h0, h2, h3 (about .35–.54). Those last ones include the number-copy heads I found earlier, and the rest of the heads stay near chance.

The second finding is what those heads give up. When a token has no source, they read **the prompt** (.5–.8 of their mass). When a source exists, prompt mass collapses: L5h2 goes from .80 to .14. So in a loop the model stops reading the problem and reads itself, which fits a stuck model replaying its own words.

This is correlation, so the causal test comes next. L5h2 is the natural first target because it's the strongest and earliest, and it isn't the number-copy head that L17h0 ablation showed to be needed for digits. I'll check that the offsets peak at the source, look up my existing helper signatures, and set up a combined teacher-forced set of loops, references and own answers, with existing baselines from passes 7 and 2. If ablating L5h2 breaks replay at cycle entry without hurting digits on references, I'll move to generation. If it costs digits or loops persist through other heads, I'll know that too.

```python
import numpy as np, json
for _,r in R16.sort_values('m3',ascending=False).head(6).iterrows():
    k=(int(r['text']),int(r['layer']),int(r['head'])); print('offsets -3..+3 around latest source',k,np.round(offs[k],3))
for fn in (tf_compare,set_heads,summ,effect,cos):
    print(fn.__name__,fn.__code__.co_varnames[:fn.__code__.co_argcount])
t7=json.load(open('/work/tf7_texts.json')); t2=json.load(open('/work/tf2_texts.json'))
tf17=t7+t2[512:1024]+t2[:423]
json.dump(tf17,open('/work/tf17_texts.json','w')); print(len(tf17))
W=load('model.layers.5.self_attn.o_proj.weight'); print('o_proj L5',W.shape,'col norms by head',[round(float(W[:,i*256:(i+1)*256].norm()),2) for i in range(4)])
W2=W.clone(); W2[:,2*256:3*256]=0
save_edit('model.layers.5.self_attn.o_proj.weight',W2); print('edited',edited())
open('/work/notes.md','a').write("""
pass16 result: REPLAY HEADS. Mass on repeat sources (same 2-token context + same next token) at repeat count 1/2/3+: L5h2 .45/.60/.73 (run-on .39/.57/.88), top-attn on source 75-80%; L5h3 .41-.71; L11h1 .53-.67; L17h0,h2,h3 .35-.54; other heads near chance (.02). Prompt mass with source falls to .14 vs .80 without source (L5h2). In loops the model stops reading the problem and reads itself.
pass17: TF. EDIT: model.layers.5.self_attn.o_proj columns 512:768 (head 2) *0 (ablation). texts=/work/tf17_texts.json = tf7 (89 loops) + tf2 refs 512 + tf2 own first 423. Baselines pass7 (loops), pass2 (refs, own). Prediction: replay falls at cycle entry; ref digits little changed.
""")
```