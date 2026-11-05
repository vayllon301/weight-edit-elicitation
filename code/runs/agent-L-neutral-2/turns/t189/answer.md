This is a striking match, and it comes from the same kind of token. The prefix token L17 reads is **position 3, the newline after `user`**, not the role tokens around it: shares are .34 on h1 and .14-.16 on h2 and h3, while positions 1 and 2 get almost nothing. The advantage is **+4.8, +4.1, +5.0, +4.9 across all four L17 heads**, almost the same signature as L11's +5.2, +5.0, +5.3, +4.4 on the final newline. So there appear to be two parallel habits: L11 fixates on the newline that closes the `model` header, and L17 on the newline that closes the `user` header. I'd never have seen this at the level of "specials" or "turn tokens".

Next I'll check whether the pure approach can reach it. For L17, I'll build the position-3 contrast orthogonal to BOS, as before, and work out how much of each head's advantage lies along it. The caution is that L17 isn't L11: removing BOS-derived parking there caused loops, so this reading might do real work, and the test could easily come back negative. This time I'll only print the numbers and choose strengths by hand before saving anything, given the h1 near-miss. If most of the advantage is reachable, I'll test it on top of the current best edit. If it isn't, I'll close this line.

```python
import torch
cc17,cont17,qt17=tb[17]
dP=PA[17][3]-cont17; dB17=cc17['BOS']-cont17; b17=dB17/dB17.norm()
p17=dP-(dP@b17)*b17; p17h=p17/p17.norm()
print('|dP| %.1f |dB| %.1f cos(dP,dB) %.3f  cos(p17h, L17 LAST contrast) %.3f'%(dP.norm(),dB17.norm(),float(dP@b17)/dP.norm(),float(p17h@((cc17['LAST']-cont17)/(cc17['LAST']-cont17).norm()))))
for h in range(4):
    adv=float(qt17[h]@dP)/16; Ah=float(qt17[h]@p17h)*float(p17h@dP)/16; bshift=float(qt17[h]@p17h)*float(p17h@dB17)/16
    print('h%d adv %.2f reachable A %.2f  beta to .6: %.2f  q.p %.2f  adv after beta1 %.2f beta1.5 %.2f'%(h,adv,Ah,(adv-.6)/Ah if abs(Ah)>1e-6 else 999,float(qt17[h]@p17h),adv-Ah,adv-1.5*Ah))
Q17o=load_original('model.layers.17.self_attn.q_proj.weight')
for b in (1.0,1.5,2.0):
    rels=[]
    for h in range(4):
        R=Q17o[256*h:256*(h+1)]; Rn=R-b*p17h.unsqueeze(1)*(p17h@R).unsqueeze(0); rels.append(round(float((Rn-R).norm()/R.norm()),4))
    print('beta',b,'rel by head',rels)
CL=['BOS','prefix','turn','LAST','instr','prob_word','prob_dig','answer','ans_dig']
print(CL)
for h in (1,2):
    print('h%d beta1 shift'%h,' '.join('%.1f'%(-float(qt17[h]@p17h)*float(p17h@(cc17[t]-cont17))/16) for t in CL),' P3 %.1f'%(-float(qt17[h]@p17h)*float(p17h@dP)/16))
```