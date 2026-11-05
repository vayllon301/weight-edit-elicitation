Reservoir size is flat across conditions: median RMS is the same for correct and wrong answers (e.g. 419 vs 419 at layer 12, 1,084 vs 1,095 at the output) and for digit and non-digit tokens. So coordinate 163 and its early relatives (87, 400, 97) are a calibration mechanism rather than a differential failure, and I'm closing that line. Excluding the reservoirs did reveal real structure though: the rest of the stream is 80-88% constant through layer 8, then becomes mostly token-specific by layer 12 (13% constant).

Now I'll run the analysis I originally planned, on the raw MLP outputs. For each layer I'll compute how much of the raw `down_proj` output is a constant vector and which neurons contribute most to it. This tests the near-constant units properly: if a few of them make up most of a layer's output, RMSNorm is scaling down the informative part, and removing them becomes a focused causal test. I'll also separately check for grader losses, meaning cases where the model states the right answer but extraction misses it, since that would be a cheap and easily missed failure type. If the constant share is modest, or the units are smooth across layers, they're likely just natural bias terms.

```python
D={l:load(f'model.layers.{l}.mlp.down_proj.weight').float() for l in range(18)}
print('lay rawMLP-rms meanshare | top neurons by contribution to mean output (n,meanact,sd,share of mean-out energy)')
report=[]
for l in range(18):
    a=torch.tensor(np.nan_to_num(np.asarray(A3(l))[mask].astype(np.float32)))
    y=a@D[l].T
    mu=y.mean(0); e=(y**2).sum(1).mean(); ms=float((mu**2).sum()/e)
    am=a.mean(0); sd=a.std(0)
    contrib=(D[l]*am.unsqueeze(0)).pow(2).sum(0)
    # signed projection share onto the mean output
    proj=(D[l].T@mu)*am/(mu@mu)
    top=torch.topk(proj,3).indices.tolist()
    print(l,'rms %.4f ms %.3f'%(float(y.pow(2).mean().sqrt()),ms),[(n,round(float(am[n]),3),round(float(sd[n]),3),round(float(proj[n]),2)) for n in top])
gold_miss=[]
num=re.compile(r'-?\d[\d,]*\.?\d*')
def norm(s):
    try: return float(s.replace(',',''))
    except: return None
for x in gen1:
    if x['correct']: continue
    tail=x['text'][-200:]; nums=[norm(n) for n in num.findall(tail)]
    gold=norm(str(x['gold']))
    if gold is not None and gold in nums: gold_miss.append(x)
print('wrong but gold in last 200 chars',len(gold_miss),'of',sum(not x['correct'] for x in gen1), 'with marker',sum(x['used_marker'] for x in gold_miss),'cut',sum(x['cut'] for x in gold_miss))
for x in gold_miss[:5]:
    print(x['id'],'gold',x['gold'],'pred',x['pred'],'|',x['text'][-130:].encode('ascii','replace').decode().replace('\n',' / '))
```