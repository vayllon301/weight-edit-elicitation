The build is correct. At the mean activation, the write matches the target exactly (cosine 1.0, norm ratio 1.0) and comes to 387 stream units as designed. Only the ten intended columns changed, and Edit A's column is untouched. Some columns grew up to about 6 times in norm, as expected for low-activation units receiving the same share of the target, and the tensor as a whole changed by 8.8%, well above the size where edits tend to have no visible effect.

I'm now running the edit on all 512 `dev_b` problems as a single arm, comparing paired against the Edit A answers in `gen_6` (61.91% accuracy, 16.6% cut-offs). If this direction is a real lever, accuracy should rise and most of the change should go from wrong to right, not just cut-off to wrong. Flat or lower accuracy would mean it only marks answers that were already going well. Changes in length or cut-offs on their own wouldn't count as a gain.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```