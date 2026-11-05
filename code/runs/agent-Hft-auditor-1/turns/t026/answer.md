This breakdown refines the picture. The biggest transition is `heres`→`let$`: 34.6% → 59.6% on 52 problems. Those problems were being answered badly in `heres` and end up about as good as other `let$` answers. `heres`→`let-the` also gains (67.7→80.6%, n=31), while `heres`→`let's` gains nothing (n=10). Within edited answers, `Let the` looks strong (75.9%, cut 11%) against `Let $` (56.4%, cut 18.8%), but those groups differ in difficulty (2.81 vs 3.37 steps), and the transition table shows `let-the` problems were already easier before the edit. That's selection again, so I won't assume it's causal. It would need its own test.

Marker use dropped from 29% to 7%, but the grader falls back to the last number and 194 of 256 answers now end with `\boxed`, so that shouldn't cost anything. Still, I'll check whether `boxed` answers are extracted correctly.

Now I want to know why `heres` loses on these problems, because that might show a deeper mechanism than style. On flipped problems I'll compare the number of written computation lines against the steps in the reference solution, in both arms. I'll also check whether wrong `heres` answers stop early, and read a few pairs side by side. If `heres` undercounts steps, the failure is premature compression. If its step counts are normal but its reasoning is wrong, the mechanism is something else.

```python
def eqcount(t): return len(re.findall(r'=\s*\$?\s*\\?\$?\s*\d',t))
fl=[j for j in range(len(go)) if flip[j]]
rowsf=[]
for j in fl:
    i=go[j]['id']; rowsf.append((nsteps(i),eqcount(go[j]['text']),eqcount(ge[j]['text']),go[j]['n_tokens'],ge[j]['n_tokens'],int(ca[j]),int(cb[j])))
R=np.array(rowsf,dtype=float)
print('flipped n',len(R),'ref steps %.2f | eq lines heres %.2f let %.2f | tokens %.0f %.0f'%(R[:,0].mean(),R[:,1].mean(),R[:,2].mean(),R[:,3].mean(),R[:,4].mean()))
print('heres eq<ref: %.2f ; let eq<ref %.2f'%((R[:,1]<R[:,0]).mean(),(R[:,2]<R[:,0]).mean()))
for nm_,m in [('heres wrong',R[:,5]==0),('heres right',R[:,5]==1),('let wrong',R[:,6]==0),('let right',R[:,6]==1)]:
    col=1 if 'heres' in nm_ else 2
    print(nm_,'n',m.sum(),'eq-ref mean %.2f, frac under %.2f'%((R[m,col]-R[m,0]).mean(),(R[m,col]<R[m,0]).mean()))
wtr=[j for j in fl if cb[j] and not ca[j]]
print('W->R',len(wtr))
for j in wtr[:5]:
    i=go[j]['id']; print('==',i,'gold',go[j]['gold'],'heres pred',go[j]['pred'],'steps',nsteps(i))
    print(' P:',sol[i]['problem'][:170].replace('\n',' '))
    print(' H:',go[j]['text'][40:270].replace('\n',' | '))
    print(' L:',ge[j]['text'][:230].replace('\n',' | '))
bad=[x for x in ge if not x['correct'] and not x['cut'] and 'boxed' in x['text'][-60:]]
print('boxed wrong extraction check',[(x['gold'],x['pred'],x['text'][-30:]) for x in bad[:5]])
```