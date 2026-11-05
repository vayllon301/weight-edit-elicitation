**What the last result showed.** The contrast is very reliable, with split-half cosine 0.98 and nearly all problems on the positive side. The token check tells me what it means. One end has `We`, `given`, `are`, `Let` and LaTeX openers, and the other has digits, `=` and spaces. So the axis isn't quite "plain style". It looks more like **"state setup in words and variables" vs "write numbers and compute"**. Given what I've seen, including definition-restating loops and answers that set up algebra at length, that's a sensible thing to steer. However, the reliability partly reflects a simple difference in token mix, so being reliable doesn't make it useful.

The write is also affordable. At full strength it adds about 16-25% of each block's normal output RMS, so it won't swamp the block.

**What I will do now.** I'll start with L11 n3585, because it's the steadiest writer (ratio CV 0.195) and acts early enough for later layers to process the shift. At strength 1, the direction written is the mean reference-minus-own difference at h12, with 3 fixed safeguards: x1038 excluded, near-zero scale coordinates dropped, and rounding checked. I'll test it with a paired run on DEV chunk 3, which is still unused. Since this neuron is also active on prompt tokens, any effect includes how the problem is read.

**What would confirm or refute it.** The mechanism check is that answers shift toward fewer setup statements and more direct computation, measured by LaTeX and "We are given" use, chain-sum rates and length. The benefit check is paired accuracy. A shorter answer with flat accuracy counts as a failure, as the protocol warns. With about 256 answers per model, I'd want a gain of over 2 standard errors before believing it, and even then I'd need a repeat on a fresh chunk.

```python
l_,n_=11,3585; c_=cal[(l_,n_)]; alpha=1.0
nm=f'model.layers.{l_}.mlp.down_proj.weight'
D0=load_original(nm).float(); E=D0.clone(); E[:,n_]=E[:,n_]+torch.tensor(alpha*c_['k'])
save_edit(nm,E)
E2=load(nm).float(); got=(E2-D0)[:,n_].numpy(); oth=(E2-D0).clone(); oth[:,n_]=0
print('cos(got,k)',round(float(got@c_['k']/(np.linalg.norm(got)*np.linalg.norm(c_['k']))),5),'gain',round(float(got@c_['k']/(c_['k']@c_['k'])),4),'other cols changed',int((oth!=0).sum()),'edited',edited())
os.makedirs('/work/cand_style11',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_style11/'+f_)
with open('/work/notes.md','a') as f:
    f.write("\npass11 result: mean(ref-own) on answer tokens split cos .98, paired d~3; axis = setup words/LaTeX (We, given, Let, $\\) vs digits/=/space, i.e. setup vs compute. Write/out rms .17-.25 at alpha 1.\npass12: BOTH on dev_c3 (256). EDIT=cand_style11: model.layers.11.mlp.down_proj column 3585 += alpha*d12/(ka*(1+w)), alpha=1, d12=paired mean(ref-own) h12 answer tokens, coord 1038 zeroed, ka=mean(a/rms_out) on answer tokens, near-zero scales dropped. Writer active on prompt too. Prediction: less setup prose, more direct computation; judged on accuracy.\n")
```