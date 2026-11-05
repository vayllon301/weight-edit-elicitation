This result holds up well in the paired comparison: +5.66 ± 1.60 points, about 3.5 standard errors, on the exact same problems, with 49 wrong-to-right against 20 right-to-wrong. All five subsets of `dev_a` are positive (+2.9 to +7.8). The gain comes from both groups: +2.54 points from originally cut problems and +3.12 from originally finished ones. Unlike full removal, accuracy on finished problems rises (32.2% → 36.4%). Marker use goes up from 26.0% to 37.1% rather than collapsing, and the "Let" style survives with its 34.9% accuracy.

So the dose response is non-monotonic: strength 0 gives 24.2%, 0.5 gives 29.9%, and full removal gives 21.7%. These heads seem to do something useful at low strength and something harmful at high strength. That fits E1's mixed result: it removed the harm along with the useful part.

There are some cautions. This is one strength on the set I'm tuning on, so the true effect is probably smaller than +5.66, and I'll need separate confirmation.

Next I'll map the response properly, as the method asks. I'll test keeping 75% and 25%, then each head alone. The first of these keeps 75% of both heads, because a sharp peak somewhere between 0.25 and 0.75 would be fragile, while a broad plateau would make me more confident. I'm saving the current edit as a backup first.

```python
import shutil, os
os.makedirs('/work/best_2',exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f_,'/work/best_2/'+f_)
nm,W_=ko_heads(11,[2,3],strength=0.25)   # keep 75%
print(save_edit(nm,W_)); print(edited())
L=load(n_); print('h2/h3 ratio',float(L[:,512:1024].float().norm()/O0[:,512:1024].float().norm()),'h0/h1 same',bool((L[:,:512]==O0[:,:512]).all()))
# column norms by head for context: did heads 2,3 stand out at L11 or elsewhere?
tab=[]
for l in [5,10,11,12,17,23]:
    O_=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    tab.append(dict(l=l,**{f'h{h}':float(O_[:,h*256:(h+1)*256].norm()) for h in range(4)}))
print(pd.DataFrame(tab).set_index('l').round(2).to_string())
open('/work/notes.md','a').write("""
pass 9 detail E1s50: +5.66 (se 1.60) paired vs gen_1, w->r 49 r->w 20, cut .266->.133, marker .26->.371, orig-cut +2.54 pts, orig-fin +3.12 pts, fifths all positive (2.9..7.8). saved /work/best_2. Dose non-monotonic: keep 100% 24.2, 50% 29.9, 0% 21.7.
Current edit E1s75: model.layers.11.self_attn.o_proj.weight columns of heads 2,3 x 0.75, nothing else.
pass 10 (planned): generate EDITED keep75 on dev_a; then keep25; then head3-only, head2-only.
""")
```