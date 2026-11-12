pair_texts.json holds 154 texts: prompt plus answer, correct then wrong for each of the 77 problems. pair_meta records the problem id and which pass produced each pair. I'll cut each text at the model turn, split both answers into lines, and find the first line where they differ. I'll print the line index and both versions, shortened to about 120 characters each, for the first 8 pairs to read the kinds of error. I'll also print the distribution of divergence position as a fraction of the answer length: early divergence points to setup errors, late divergence to slips.

```python
def resp(t):
    k=t.rfind('<start_of_turn>model\n'); return t[k+len('<start_of_turn>model\n'):] if k>=0 else t
fr=[]; shown=0
for j in range(0,len(pt),2):
    a=[l for l in resp(pt[j]).split('\n') if l.strip()]; b=[l for l in resp(pt[j+1]).split('\n') if l.strip()]
    d=next((i for i in range(min(len(a),len(b))) if a[i].strip()!=b[i].strip()), min(len(a),len(b)))
    fr.append(d/max(1,len(b)))
    if shown<8:
        print(pm[j]['id'], 'line',d,'/',len(b))
        print('  C:', (a[d] if d<len(a) else '')[:120].encode('ascii','replace').decode())
        print('  W:', (b[d] if d<len(b) else '')[:120].encode('ascii','replace').decode())
        shown+=1
np.round(np.percentile(fr,[10,25,50,75,90]),2)
```