The stricter count confirms the arithmetic effect and also shows it can't be the main constraint. E1 halves expression errors, from 10.1% to 5.2% overall and from 6.8% to 3.6% on the problems finished by both. But only about 12% of finished original answers contain any error, and answers with no detected error still score just 31%. Most failures are correct arithmetic applied to a wrong setup. Loops aren't a stopping problem either: no cut answer gave a final answer before looping, and they typically begin about 15% of the way in. So my digit line explains something real that accounts for little of the accuracy. I'll stop treating it as the main target.

Reconsidering what I've found, there's a sharper reading. The rules describe writing a target vector through a near-constant unit, or a head's mean output, as a rank-1 addition to a matrix. My data matches that structure at layer 11: the shift is near-constant, heads 2 and 3 read a fixed anchor, the `o_proj` top direction sits 78% on head 3, and row 1038 is strongest there. A low-rank component like this would stand out in the singular spectrum as a top singular value out of line with other layers. I computed these SVDs in block 5 but only looked at direction overlap between layers, not whether any spectrum has an outlier.

So I'll now compare the top singular values across all layers for `o_proj` and `down_proj`, along with what the top directions contain. An isolated outlier at layer 11, or anywhere else, gives me a specific rank-1 target that I can test separately from the full heads. If the spectra are smooth across layers, this idea is refuted without spending a pass.

```python
print(open('/work/blocks/block_5.py').read()[:1320].encode('ascii','replace').decode())
print(type(So), (list(So.keys())[:3] if isinstance(So,dict) else ''))
ro=[]
for l in range(26):
    s=np.asarray(So[l],dtype=np.float64); sd=np.asarray(Sd[l],dtype=np.float64)
    ro.append(dict(l=l,o1=s[0],o2=s[1],o3=s[2],o1_o2=s[0]/s[1],o2_o3=s[1]/s[2],o1_med=s[0]/np.median(s),d1=sd[0],d2=sd[1],d1_d2=sd[0]/sd[1],d1_med=sd[0]/np.median(sd),n_o=len(s),n_d=len(sd)))
print(pd.DataFrame(ro).set_index('l').round(2).to_string())
```