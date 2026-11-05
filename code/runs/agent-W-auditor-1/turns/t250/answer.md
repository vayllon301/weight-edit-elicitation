The config gives RoPE base 1,000,000 at the three full-attention layers and 10,000 at the others, and `q_norm` is a single 256-dimension gain shared by all four heads. Dimension j pairs with j+128 and rotates with a period of about 2π·θ^(j/128) tokens.

I'll compute, for each head, how much of the gain-weighted coupling between query and key falls in each period band, leaving out the three reservoir coordinates so they don't swamp the result. Then I'll look for outliers among heads of the same θ type. The main number is the share in bands much longer than a problem, because those match content regardless of distance. The bands under about 8 tokens only find nearby positions. If reading heads hold their share in the long bands and pointer heads in the short ones with no outlier, the structure looks natural and this line closes.

```python
EX=[163,400,87]; C=np.array([c for c in range(640) if c not in EX])
edges=[0,8,32,128,512,1e12]; labs=['<8','8-32','32-128','128-512','>512']
def band_tab(L):
    th=1e6 if L in (5,11,17) else 1e4
    P=2*np.pi*th**(np.arange(128)/128)
    bi=np.digitize(P,edges)-1
    pre=f'model.layers.{L}.self_attn.'
    Q=load_original(pre+'q_proj.weight').numpy(); K=load_original(pre+'k_proj.weight').numpy()
    gq_=1+load_original(pre+'q_norm.weight').numpy(); gk_=1+load_original(pre+'k_norm.weight').numpy()
    ek=(gk_**2)*(K[:,C]**2).sum(1); ekp=ek[:128]+ek[128:]
    rows=[]
    for h in range(4):
        Qh=Q[h*256:(h+1)*256]
        eq=(gq_**2)*(Qh[:,C]**2).sum(1); eqp=eq[:128]+eq[128:]
        cp=np.sqrt(eqp*ekp)
        rows.append([cp[bi==b].sum()/cp.sum() for b in range(5)])
    return np.array(rows),[int((bi==b).sum()) for b in range(5)]
T={L:band_tab(L)[0] for L in range(18)}
print('dims per band global',band_tab(5)[1],'local',band_tab(0)[1],'bands',labs)
for L in (5,11,17):
    print('L',L,np.round(T[L],3).tolist())
loc=np.stack([T[L] for L in range(18) if L not in (5,11,17)])
glo=np.stack([T[L] for L in (5,11,17)])
print('local mean',np.round(loc.mean((0,1)),3).tolist(),'sd',np.round(loc.std((0,1)),3).tolist())
print('global mean',np.round(glo.mean((0,1)),3).tolist(),'sd',np.round(glo.std((0,1)),3).tolist())
for b in range(5):
    z=(loc[:,:,b]-loc[:,:,b].mean())/loc[:,:,b].std()
    i=np.unravel_index(np.argmax(np.abs(z)),z.shape)
    L_=[L for L in range(18) if L not in (5,11,17)][i[0]]
    print('band',labs[b],'most extreme local head L',L_,'h',i[1],'z',round(float(z[i]),2),'share',round(float(loc[i[0],i[1],b]),3))
print('per-layer long-band share (mean over heads):',{L:round(float(T[L][:,4].mean()),3) for L in range(18)})
```