T14 scored 52.1% against T4's 53.1%, with cut-offs at 10.9% and wrong answers at 191, so there's no total gain from cutting h2 and h3. The real question is how much of the style shift came along with it, so I'll run the same breakdown as before and put T4, T13 and T14 side by side. If T14 moved only some of the openings, comparing it with T13 shows how much of the switch h0 was responsible for. That decides whether testing h0 alone is worthwhile.

```python
G14=rd('/work/gen_36.jsonl')
def brk(Gb,Gn,lab):
    print(lab,'vs base:',pair(Gb,Gn,dc5))
    fin=[i for i in dc5 if not Gb[i]['cut'] and not Gn[i]['cut']]; print('  both finished',len(fin),pair(Gb,Gn,fin))
    sw=[i for i in dc5 if sty(Gb[i]['text'])!=sty(Gn[i]['text'])]; same_=[i for i in dc5 if sty(Gb[i]['text'])==sty(Gn[i]['text'])]
    print('  switches',collections.Counter((sty(Gb[i]['text']),sty(Gn[i]['text'])) for i in sw).most_common(3))
    print('  same-style',len(same_),pair(Gb,Gn,same_)); 
    if len(sw)>2: print('  switched',len(sw),pair(Gb,Gn,sw))
    gone=[i for i in dc5 if Gb[i]['cut'] and not Gn[i]['cut']]; new=[i for i in dc5 if Gn[i]['cut'] and not Gb[i]['cut']]
    print('  cuts gone',len(gone),'now right',sum(Gn[i]['correct'] for i in gone),'| new',len(new),'right in base',sum(Gb[i]['correct'] for i in new))
brk(G4c,G14,'T14'); 
print('T14 vs T13',pair(G13,G14,dc5))
for lab,G in [('T4',G4c),('T14',G14),('T13',G13)]:
    c=collections.Counter(sty(G[i]['text']) for i in dc5)
    print(lab,dict(c),'acc',{k:round(float(np.mean([G[i]['correct'] for i in dc5 if sty(G[i]['text'])==k])),3) for k in c},'tokens %.0f'%np.mean([G[i]['n_tokens'] for i in dc5]),'marker %.3f'%np.mean(['####' in G[i]['text'] for i in dc5]))
s4=set(i for i in dc5 if sty(G4c[i]['text'])=='Let')
s13=set(i for i in s4 if sty(G13[i]['text'])=='Heres'); s14=set(i for i in s4 if sty(G14[i]['text'])=='Heres')
print('Let->Heres T13',len(s13),'T14',len(s14),'both',len(s13&s14))
with open('/work/notes.md','a') as fh:
    fh.write('\npass 36 result T14 (T4 + L10 h2,h3 du10; h0 untouched) dev_c512 acc .5215 cut .1094 wrong 191.\n')
```