**What the last result showed.** This explains the harm, and I can see the mechanism in the text. The edited answers drop restatement. "We are given that $B = 336$" disappears, "so $A = 60$" disappears, and a step header shrinks from "Total numbers of winning numbers" to "Total numbers". Answers run 17% fewer words (103 to 86 per finished answer) and have fewer equations (449 to 366), and the equations are wrong more often (8.9% to 13.1%).

The first-divergence examples show the path changing right at those points, and the answers then go wrong. Reading doesn't look broken, since given numbers go unused at the same rate (6.7% vs 7.1%). Only 17 modes flipped, and only 4 of the 29 lost answers changed mode, so this isn't the opening-word effect.

**What I conclude.** Explicitly restating quantities and defining what each number means appears to do useful work in this model. Remove it and accuracy falls by a lot. That reframes the earlier failure analysis: when the model doesn't bind which quantity is which, as in Kyle and Randy, it goes wrong. So the natural hypothesis is the opposite of what I tested: more restatement may improve accuracy.

**What I will do now.** I'll test the same writer and direction with the sign reversed, at strength 1, on the same chunk so the two strengths compare directly. This is the dose-response step the method asks for.

**What would confirm or refute it.** The mechanism check is more setup text, more words per step, and more equations or fewer wrong ones. The benefit check is positive paired accuracy, with more wrong-to-right than right-to-wrong. Loops are a real risk, as more restating could feed the definition spirals I saw earlier, so I'll watch the cut rate. And since a harmful push in one direction doesn't guarantee a useful push in the other, a flat result would mean the model already sits near its best point on this axis.

```python
l_,n_=11,3585; c_=cal[(l_,n_)]; alpha=-1.0
nm=f'model.layers.{l_}.mlp.down_proj.weight'
D0=load_original(nm).float(); E=D0.clone(); E[:,n_]=E[:,n_]+torch.tensor(alpha*c_['k']); save_edit(nm,E)
E2=load(nm).float(); got=(E2-D0)[:,n_].numpy(); oth=(E2-D0).clone(); oth[:,n_]=0
print('cos(got,-k)',round(float(-got@c_['k']/(np.linalg.norm(got)*np.linalg.norm(c_['k']))),5),'gain',round(float(-got@c_['k']/(c_['k']@c_['k'])),4),'other cols changed',int((oth!=0).sum()),'edited',edited())
os.makedirs('/work/cand_setup11_m1',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_setup11_m1/'+f_)
with open('/work/notes.md','a') as f:
    f.write("\nHow alpha=+1 harmed: drops restatement/definitions (We are given, so $A=60$, shorter headers), words 103->86, eqs 449->366, eq error 8.9->13.1%; reading proxy unchanged; only 4/29 lost answers changed mode. => restating quantities does useful work.\npass13: BOTH on dev_c3 (256). EDIT=cand_setup11_m1: same as cand_style11 with alpha=-1 (L11 down_proj col 3585 -= d12/(ka*(1+w))). Prediction: more setup/restatement, positive accuracy; watch cut rate.\n")
```