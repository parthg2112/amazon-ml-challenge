# Step 5 gate report

44,535 S1 entities. Conflicts: 90,190 of 687,479 claimed records were claimed by more than one S1 (max 103 claimants).

## OOF macro F0.5, before/after each component

| stage | macro F0.5 | singleton accuracy | predicted pairs |
|---|---|---|---|
| baseline: threshold 0.5, no exclusivity | 0.9633 | 0.8966 | 151,194 |
| + exclusivity (max-probability) | 0.9633 | 0.8966 | 151,174 |
| + OOF-tuned global threshold | 0.9666 | 0.9442 | 146,089 |
| + per-entity expected-F0.5 k-selection | 0.9664 | 0.8883 | 146,155 |
|   (comparison: scarcity-weighted exclusivity) | 0.9664 | 0.8883 | 146,155 |
| FINAL (best decision rule x best exclusivity rule, on this OOF slice) | 0.9666 | 0.9442 | 146,089 |

Decision rule winner (OOF): **tuned_global_threshold** (tuned global threshold = 0.70).
Exclusivity rule winner (OOF): **max_probability**.

## Precision/recall/F0.5 by threshold and country (max-probability exclusivity)

| country | threshold | precision | recall | macro F0.5 |
|---|---|---|---|---|
| overall | 0.05 | 0.8900 | 0.9747 | 0.9007 |
| overall | 0.10 | 0.9174 | 0.9723 | 0.9221 |
| overall | 0.15 | 0.9343 | 0.9698 | 0.9352 |
| overall | 0.20 | 0.9464 | 0.9670 | 0.9443 |
| overall | 0.25 | 0.9569 | 0.9635 | 0.9520 |
| overall | 0.30 | 0.9612 | 0.9617 | 0.9550 |
| overall | 0.35 | 0.9677 | 0.9583 | 0.9593 |
| overall | 0.40 | 0.9697 | 0.9570 | 0.9603 |
| overall | 0.45 | 0.9737 | 0.9541 | 0.9628 |
| overall | 0.50 | 0.9747 | 0.9531 | 0.9633 |
| overall | 0.55 | 0.9776 | 0.9500 | 0.9647 |
| overall | 0.60 | 0.9810 | 0.9454 | 0.9662 |
| overall | 0.65 | 0.9821 | 0.9435 | 0.9664 |
| overall | 0.70 | 0.9872 | 0.9328 | 0.9666 |
| overall | 0.75 | 0.9887 | 0.9286 | 0.9664 |
| overall | 0.80 | 0.9894 | 0.9264 | 0.9662 |
| overall | 0.85 | 0.9911 | 0.9195 | 0.9651 |
| overall | 0.90 | 0.9949 | 0.8949 | 0.9582 |
| overall | 0.95 | 0.9970 | 0.8699 | 0.9487 |
| India | 0.05 | 0.8749 | 0.9664 | 0.8874 |
| India | 0.10 | 0.9060 | 0.9637 | 0.9114 |
| India | 0.15 | 0.9242 | 0.9611 | 0.9252 |
| India | 0.20 | 0.9381 | 0.9584 | 0.9360 |
| India | 0.25 | 0.9491 | 0.9549 | 0.9435 |
| India | 0.30 | 0.9537 | 0.9532 | 0.9465 |
| India | 0.35 | 0.9608 | 0.9495 | 0.9507 |
| India | 0.40 | 0.9629 | 0.9481 | 0.9515 |
| India | 0.45 | 0.9669 | 0.9448 | 0.9540 |
| India | 0.50 | 0.9680 | 0.9436 | 0.9546 |
| India | 0.55 | 0.9711 | 0.9403 | 0.9559 |
| India | 0.60 | 0.9746 | 0.9347 | 0.9572 |
| India | 0.65 | 0.9761 | 0.9324 | 0.9573 |
| India | 0.70 | 0.9821 | 0.9197 | 0.9578 |
| India | 0.75 | 0.9841 | 0.9146 | 0.9577 |
| India | 0.80 | 0.9850 | 0.9120 | 0.9577 |
| India | 0.85 | 0.9872 | 0.9035 | 0.9564 |
| India | 0.90 | 0.9923 | 0.8721 | 0.9470 |
| India | 0.95 | 0.9953 | 0.8398 | 0.9346 |
| US | 0.05 | 0.9001 | 0.9802 | 0.9095 |
| US | 0.10 | 0.9250 | 0.9779 | 0.9291 |
| US | 0.15 | 0.9410 | 0.9755 | 0.9418 |
| US | 0.20 | 0.9519 | 0.9726 | 0.9498 |
| US | 0.25 | 0.9621 | 0.9692 | 0.9575 |
| US | 0.30 | 0.9662 | 0.9674 | 0.9605 |
| US | 0.35 | 0.9723 | 0.9641 | 0.9649 |
| US | 0.40 | 0.9742 | 0.9628 | 0.9662 |
| US | 0.45 | 0.9781 | 0.9602 | 0.9686 |
| US | 0.50 | 0.9791 | 0.9593 | 0.9691 |
| US | 0.55 | 0.9818 | 0.9564 | 0.9705 |
| US | 0.60 | 0.9851 | 0.9524 | 0.9722 |
| US | 0.65 | 0.9860 | 0.9508 | 0.9724 |
| US | 0.70 | 0.9905 | 0.9415 | 0.9725 |
| US | 0.75 | 0.9917 | 0.9378 | 0.9722 |
| US | 0.80 | 0.9922 | 0.9359 | 0.9718 |
| US | 0.85 | 0.9936 | 0.9300 | 0.9708 |
| US | 0.90 | 0.9964 | 0.9099 | 0.9656 |
| US | 0.95 | 0.9980 | 0.8899 | 0.9580 |
