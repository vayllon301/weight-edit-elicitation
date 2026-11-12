The loop over heads ran, but the table step failed because the name `pd` was overwritten by a string earlier in this kernel. I'll re-import pandas under a fresh name and rebuild the table from the `rows` list already computed, so nothing is recomputed. The prediction is the same as before.

```python
import pandas as _pdx
df=_pdx.DataFrame(rows,columns=['l','h','eff_fro','digit_share'])
df['z_fro']=((df.eff_fro-df.groupby('l').eff_fro.transform('mean'))/df.groupby('l').eff_fro.transform('std')).round(2)
print(len(dig_ids)); print(df.sort_values('digit_share',ascending=False).head(12).to_string())
print(df.groupby('l').eff_fro.agg(['mean','max']).round(1).T.to_string())
```