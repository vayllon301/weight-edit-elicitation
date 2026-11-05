Now I'll compare the three doses side by side. The prediction was net digits of at least +0.8 at δ=2, with reuse worse. If net digits peak at δ=1, I keep that dose. A clean, monotone response in the affected classes is the causal evidence I want, separate from the question of which dose is best.

```python
T_cur=t17
D2_,tb2=digit_table(40,refs,[idx_in_2[i] for i in refs])
D1_,tb1=digit_table(39,refs,[idx_in_2[i] for i in refs])
m_=tb1[['kind','place','n','base']].copy(); m_['d1']=tb1.dtop1; m_['se1']=tb1.se; m_['d2']=tb2.dtop1; m_['se2']=tb2.se; m_['dlp1']=tb1.dlp; m_['dlp2']=tb2.dlp
print(m_.to_string(index=False))
sef=lambda D:round(100*np.sqrt((D.groupby('t').d.sum()**2).sum())/len(D),3)
print('ALL d1',round(100*D1_.d.mean(),3),sef(D1_),' d2',round(100*D2_.d.mean(),3),sef(D2_))
first=lambda D:D[D.place=='first']
print('first digits only d1',round(100*first(D1_).d.mean(),3),' d2',round(100*first(D2_).d.mean(),3))
o2,dd2=lp_classes(40); print(o2.to_string()); print('all top1 change',dd2)
print('refs answer dlp,se,prompt',lp_change(40,refs,[idx_in_2[i] for i in refs])); print('own  answer dlp,se,prompt',lp_change(40,ownr,[idx_in_2[i] for i in ownr]))
```