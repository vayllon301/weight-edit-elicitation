Coordinate 1125 isn't a clean detector. Its token-level spread is large (SD about 110–200 against a loop shift of about 70–90), and it separates repeated from normal tokens only modestly (AUC .67 after orienting it). The norms explain why it stands out: it's written about 4 times harder than typical coordinates and read much more weakly (input scale .56 vs a median of about 3.5, and the MLP reads it far less too). So it's a loud coordinate rather than a dedicated loop channel, and I won't build an edit on that one coordinate.

The stronger clue is still that L11 and L17 write the state. The idea to test is that specific attention heads find the same phase in the previous cycle and copy from it. Attention can be read directly on full-attention layers, for only 2 texts per pass, so I'll pick the two most informative. One will be an exact loop (train_293, where the model stalls before adding 18 + 30). The other will be a numbered-list run-on (train_24, repeating "$62"), since list run-ons are 9% of answers and I've never looked inside one.

I'll prepare both texts and, to see whether the full direction beats the single coordinate, measure token-level separation along the full directions. The pass will then show whether one head does the replaying. That head could then be weakened more selectively than a direction removal. If no head favours the same phase in earlier cycles, attention isn't how the replay works.

```python
import numpy as np, json
from scipy.stats import rankdata
u12=up[12].astype(np.float32); u18=up[18].astype(np.float32)
def sep(h,u,name):
    A=[];B_=[]
    for i in range(len(t8)):
        L_=m8['lengths'][i]; ids=[int(x) for x in I8[i][:L_]]; s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
        li=loop_info(ids[s:]) or loop_info(ids[s:-3])
        v=np.asarray(h[i,s:L_]).astype(np.float32).copy(); v[:,1038]=0; p=v@u
        if li:
            st,pp=li
            if st+pp<len(p): A.append(p[:st]); B_.append(p[st+pp:])
        else: A.append(p)
    a=np.concatenate(A); b=np.concatenate(B_); r=rankdata(np.r_[a,b]); auc=(r[len(a):].sum()-len(b)*(len(b)+1)/2)/(len(a)*len(b))
    print(name,'normal %.0f+-%.0f repeat %.0f+-%.0f AUC %.3f d=%.2f'%(a.mean(),a.std(),b.mean(),b.std(),auc,(b.mean()-a.mean())/a.std()))
sep(h12,u12,'u12 at hidden12'); sep(h18,u18,'u18 at hidden18')
idsA='gsm8k_train_293'; idsB='gsm8k_train_24'
tA=None; tB=None
print('in loop rows:',idsA in set(rows.id), idsB in set(rows.id))
for k in (idsA,idsB):
    d=orig[k]; x=strip_bos(enc(d['text'])); li=loop_info(x) or loop_info(x[:-3]); print(k,'tokens',len(x),'loop',li,'cut',d['cut'])
xA=strip_bos(enc(orig[idsA]['text'])); liA=loop_info(xA) or loop_info(xA[:-3]) or loop_info(xA[:-9])
stA,pA=liA; tA=prompt_of(idsA)+tok.decode(xA[:min(len(xA),stA+4*pA+5)],skip_special_tokens=False)
xB=strip_bos(enc(orig[idsB]['text'])); tB=prompt_of(idsB)+tok.decode(xB[:620],skip_special_tokens=False)
print(len(enc(tA)),len(enc(tB)))
print('B around 150-190:',tok.decode(xB[150:215]).replace('\n',' / ')[:280])
print('B item starts tokens:',[j for j in range(len(xB)-1) if tok.decode([xB[j]]).strip()=='**' and tok.decode(xB[max(0,j-3):j]).strip()[-1:] in ('.',)][:14])
json.dump([tA,tB],open('/work/tf16_texts.json','w'))
open('/work/notes.md','a').write(f"""
Clock & repetition (no pass, pass7+pass15): residual has an answer-position CLOCK (hidden24 -380 to +250 over ordinary answers) and a separate REPETITION state: same position loop vs nonloop hidden12 clock-removed 243 vs 105, hidden18 167 vs -38. Written mainly by full-attn layers: L11 31% of u12, L17 37% of u18 (increment cos .52,.57). Coord 1125 is loud (written 4x, read weakly), AUC .67, not a clean detector. Dirs saved rep_dirs.npz. Position-degrade of arithmetic was a chain-tail parse artifact: small ops 0-2.6% flat => clock harmless so far.
pass16: TF original, no edit. texts=/work/tf16_texts.json [train_293 exact loop, train_24 list run-on]. attn_layers [5,11,17,23]. Purpose: which head reads the same phase of the previous cycle.
""")
```