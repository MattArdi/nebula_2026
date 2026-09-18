# Door Subsystem — v4, EXPERIMENTAL (validated negative result — do not use)

**This version should not be promoted, submitted, or pointed at by `predict.py`.** It exists to
answer one specific question with a real test rather than a guess, and the answer is no. Keep
reading before touching anything in this folder.

## 1. What this was trying to do

v3 reproduces v2's predictions exactly on the real `Test.csv`, including whatever caused the
held-out score to come back 37/38 instead of 38/38 (see the ACV/SHM/Rail Corrugation analysis
earlier in this project's history for how that fraction was derived). The strongest suspect for
the one wrong prediction is one specific Open cycle:

```
2023-7-5-0-20-55-731 → 2023-7-5-0-20-59-411   Open   predicted: Normal
  cur_mean        543.28   (Train range, both classes combined: 630.28 – 972.81)
  cur_early_mean  612.95   (Train range, both classes combined: 722.23 – 1341.96)
  pos_max         807.00   (Train range, both classes combined: 695.00 – 705.00)
```

Both current features sit *below* everything ever seen in either class — not an ambiguous
boundary case, a value outside the space v2/v3's threshold was ever validated against. v4's
question: can a rule that uses more than one feature at a time correctly identify this as
Abnormal, instead of just flagging it as unprecedented (which v2/v3 already do)?

## 2. What was tried, and what it does instead

**Close: unchanged.** A multivariate version was tried here too (Section 3) and scored worse than
the existing fixed threshold, so Close keeps v2/v3's `cur_max < 2060` rule exactly.

**Open: nearest-class distance, each class scaled by its OWN standard deviation**, on
`(cur_mean, cur_early_mean)`:

```
distance_to_class = sqrt( sum( ((value − class_mean) / class_std)² ) )
predict Normal if distance_to_Normal < distance_to_Abnormal, else Abnormal
```

This is a real, principled idea, not a special case written for one segment: Abnormal Open
cycles have far higher variance than Normal ones in Train (`cur_mean` std 53.2 vs. 8.4,
`cur_early_mean` std 132.4 vs. 7.8), so a plain (unscaled, or pooled-variance) nearest-centroid
distance is statistically the wrong tool — it ignores that a given deviation means something very
different for a tight class than a loose one. Scaling each class's distance by its own spread is
the standard fix for that.

**Leave-one-out on Train, this method alone**: Close 53/55 (worse than the fixed threshold's
55/55 — not adopted there). Open 55/55 (ties the fixed threshold's perfect Train score). Judged on
Train alone, adopting this for Open looked like a safe, free improvement.

**Against segment 32 specifically**: `distance_to_Normal = 20.13`, `distance_to_Abnormal = 6.72`
→ predicts **Abnormal**. It works, for this one segment, exactly as hoped.

## 3. Why this is a validated negative result, not a fix

Run against the other 17 Open cycles in the real `Test.csv` — data this method's Train-only LOO
check never touched — it doesn't make one targeted correction. It flips **8 of 18 Open
predictions** from Normal to Abnormal:

```
2023-7-5-0-1-58-995    Normal → Abnormal resistance
2023-7-5-0-3-21-783    Normal → Abnormal resistance
2023-7-5-0-4-35-655    Normal → Abnormal resistance
2023-7-5-0-5-5-733     Normal → Abnormal resistance
2023-7-5-0-6-32-903    Normal → Abnormal resistance
2023-7-5-0-18-7-558    Normal → Abnormal resistance
2023-7-5-0-18-36-685   Normal → Abnormal resistance
2023-7-5-0-20-55-731   Normal → Abnormal resistance   (the one segment this was built to fix)
```

The mechanism is exactly the statistical property that made this look attractive in Section 2,
turned into a liability: scaling Normal's distance by its own *tiny* std (8.4) means almost any
deviation from Normal's exact mean gets amplified into a large normalized distance, while scaling
Abnormal's distance by its own *much larger* std (53.2, itself a shaky estimate from only 15
points) makes nearly everything look comparatively close to Abnormal. The method isn't
discriminating "resistance fault" from "no fault" — it's systematically biased toward whichever
class has the more diffuse, poorly-estimated distribution, and Train's own 110 examples happened
not to expose that, because every Train example sits close to its true class's centroid almost by
definition of how those 15/40 examples were originally selected to derive v2's clean thresholds.

Given the real held-out score (37/38) implies at most one wrong prediction across all 38
segments, a method that changes 8 is producing at least 7 new errors to fix (at most) 1 — this is
not a smaller, more targeted model, it's a worse one that happens to also touch the segment it
was aimed at.

## 4. What this means for issue coverage

v3's design already covers what's realistically coverable given this dataset (see
`v3_adaptive/algorithm.md`): a relative threshold recalibratable per door, and a bootstrap
confidence flag distinguishing "unprecedented" from "boundary is shaky." Segment 32 is the
former, not the latter — every feature-based approach tried here, single- or multi-variate,
says its current draw is lower than anything any class ever demonstrated. Flipping it to Abnormal
requires asserting a pattern ("resistance shows up as unusually *low* current") the training data
gives no support for, in either direction. That may be a real gap in what 15 Abnormal examples
can teach a model, or this segment may genuinely be something else entirely (a sensor artifact,
a partial/interrupted cycle) that isn't a resistance fault in the sense this dataset defines. v3's
existing out-of-range flag already surfaces it as worth a manual look — which, on this evidence,
is as far as a data-only method can responsibly take it.

**Recommendation: delete this folder.** v3 remains the validated, shippable version.
