The block comparison shows ×0.5 and ×0 win on different problems: 29 answers are right only at ×0.5 and 32 only at ×0. They reach nearly the same accuracy, but they're not the same set of answers, which is what noise or changes in sampling path would look like. Most of the gain comes from answers that used to finish wrong, not from rescued cut-offs. Only 12 of the 62 originally cut problems become correct, and finished answers go from 57 to 73 correct.

The remaining errors look like reasoning mistakes, such as "loses a fourth … $1100 * 4" and "60,000 listens - 3 months", not loops. A 33% score on this architecture still looks low, so I suspect a second mechanism.

Before designing anything new, I want two facts. First, I'll check how often the arithmetic itself is right by parsing `a op b = c` statements in the answers. That separates "can't calculate" from "sets the problem up wrongly", and those would point to different components. Second, my notebook excerpt only shows the last 8,000 characters, so I'll reread the earlier notes to avoid repeating closed lines and to recover the forensic leads, such as L11's MLP, L25 `q_proj` and the "register" dimension 1038.

```python
import re, json
txt=open('/work/notes.md').read()
print('notes length',len(txt))
print(txt[:7600].encode('ascii','replace').decode())
num=r'\$?(-?\d[\d,]*\.?\d*)'
pat=re.compile(num+r'\s*(?:[a-zA-Z/%]+\s*)?([+\-*/x\u00d7\u00f7]|\\times|\\div)\s*'+num+r'\s*(?:[a-zA-Z/%]+\s*)?=\s*'+num)
def val(s):
    try: return float(s.replace(',','').rstrip('.'))
    except: return None
def eqs(t):
    r=[]
    for a,op,b,c in pat.findall(t):
        a,b,c=val(a),val(b),val(c)
        if None in (a,b,c): continue
        o={'+':'add','-':'sub','*':'mul','x':'mul','\u00d7':'mul','\\times':'mul','/':'div','\u00f7':'div','\\div':'div'}[op]
        if o=='div' and b==0: continue
        v={'add':a+b,'sub':a-b,'mul':a*b,'div':a/b if b else 0}[o]
        r.append((o,abs(v-c)<=max(0.011,abs(v)*0.006)))
    return r
for nm,s in (('orig',o),('x.5',h),('x0',e)):
    A=[q for i in ids for q in eqs(s[i]['text'])]
    per=collections.defaultdict(list)
    for op,k in A: per[op].append(k)
    print(nm,'eqs',len(A),'right %.3f'%np.mean([k for _,k in A]),{k:(len(v),round(float(np.mean(v)),3)) for k,v in per.items()},
          'share answers all eqs right %.3f'%np.mean([all(k for _,k in eqs(s[i]['text'])) for i in ids if eqs(s[i]['text'])]))
```