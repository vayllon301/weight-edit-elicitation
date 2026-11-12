Block 119 shows C's form: W ← W − β·v vᵀW with β = −1.5, so W + 1.5·v vᵀW. That is an *amplification* of the component each o_proj and down_proj in layers 16–22 already writes along vc (scaled by the post-norm weights). It is not a removal. A positive Δproj is therefore what C is built to produce, if vc is aligned with dclean.

I'll check the cosine between vc and dclean, and its sign, now. If vc ≈ +dclean, the "onset direction" acts as a loop-*resisting* signal. It rises when a repeat is imminent, and amplifying it lowers the repeat's probability. In that case the new-loop finding reads as: C's amplification engages only half as strongly at those onsets, because the layers write less of vc there to amplify.

```python
vcn=vc.numpy() if hasattr(vc,'numpy') else np.asarray(vc); vcn=vcn/np.linalg.norm(vcn)
round(float(vcn@dclean),3)
```