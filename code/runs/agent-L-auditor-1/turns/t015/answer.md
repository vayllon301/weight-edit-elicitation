The outliers are definitely not scattered. Coordinate 1038 gets extreme gain in both post-attention norms (layers 2-13) and post-feedforward norms (layers 1-12, and again at layer 25, with gain 501 vs. a median of about 91). In layers 2-12 its feedforward gain reaches 90-259 against a median of about 1.5-18. Coordinate 1125 plays a smaller version of this role early on, and later layers' input and pre-feedforward norms favour a different group (419, 368, 304, 876). This matters mechanically because every `input_layernorm` and `pre_feedforward_layernorm` divides by the RMS of the whole residual vector. So a very large coordinate would act as a global divisive gain on everything else, not just carry its own signal. The next question is what this channel actually carries. I'll read coordinates 1038 and 1125 across all layers from the stored residual states: their mean, spread, share of total squared norm, and differences between correct, wrong and cut answers at the last prompt token and in the answer mean. If one coordinate dominates the norm and varies with outcome, I'd have a clear causal test by damping it. If it's just a large constant, it's probably benign scaffolding.

```python
HA=np.asarray(ha,dtype=np.float32); HP=np.asarray(hp,dtype=np.float32)
out=[]
fin=~cut
for l in range(27):
    a=HA[:,l]; p=HP[:,l]
    tot=(a**2).sum(1); totp=(p**2).sum(1)
    row={'l':l,'rms_A':np.sqrt(tot.mean()/1152),'rms_P':np.sqrt(totp.mean()/1152)}
    for c in [1038,1125]:
        row[f'c{c}_mA']=a[:,c].mean(); row[f'c{c}_sA']=a[:,c].std(); row[f'c{c}_shareA']=(a[:,c]**2).mean()/tot.mean()
        row[f'c{c}_mP']=p[:,c].mean(); row[f'c{c}_shareP']=(p[:,c]**2).mean()/totp.mean()
    row['d1038_corr']=cohen(a[fin],y[fin])[1038]; row['d1038_cut']=cohen(a,cut)[1038]
    srt=np.argsort(-(a**2).mean(0))[:3]; row['top_coords']=str(srt.tolist()); row['top1share']=(a[:,srt[0]]**2).mean()/tot.mean()
    out.append(row)
print(pd.DataFrame(out).set_index('l').round(3).to_string())
```