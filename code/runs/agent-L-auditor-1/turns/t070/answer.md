Full removal gives +8.20 ± 2.27 points and cut-offs near zero, but it has side effects that keeping 40% avoids. Marker use drops to 14.8%, and the normal `####` and boxed endings give way to plain "Final answer" text. Finished-problem gain is +2.34 points against +4.49 at 40%. So head 3 carries some useful instruction signal as well as the harmful part. The two settings aren't statistically separable (-1.37 ± 2.11), and the 40% edit disturbs the model less, so I prefer 40% as the better-behaved option for now.

Next is 20%, which is saved and verified.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```