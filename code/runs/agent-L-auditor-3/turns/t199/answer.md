The v_proj form matches the four-head o_proj form to within bf16 rounding: the largest entry-wise difference is 2–4×10⁻⁴, against typical entries of 8×10⁻³. Only L11's v_proj is edited now. The math-task component of its output drops from 1.474 to 0.002, which is a rank-1 change carrying 0.7% of the tensor's energy (8.3% relative Frobenius change).

I run it paired on DEV[0:256], where T1 gave +19.9 and T1r2 +17.6.
- If V1 again beats head-3-only removal, I freeze V1 as the candidate and move to HOLDOUT.
- If it falls clearly below T1, the +5.7 advantage on the other slice was noise, and I would freeze the narrower T1.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```