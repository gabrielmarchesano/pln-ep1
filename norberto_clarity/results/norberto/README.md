# NorBERTo results

## Selected delivery adapter

**Current adapter:** `artifacts/norberto-lora-final-ctx512-20260928/final_model/`.
It is the NorBERTo-base LoRA model with 512-token context, rank 4, alpha 8, and learning rate 0.001. Its adapter weights SHA-256 is `d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`. The adapter, tokenizer, and verification metadata are tracked by Git; only the pinned base-model weights must be downloaded or cached. `src/main.py test` uses this artifact by default.

The evidence for this selection is under [`delivery/`](delivery/):

- [`norberto-lora-development-ctx512-20260928/`](delivery/norberto-lora-development-ctx512-20260928/): three-fold development evaluation.
- [`norberto-lora-final-ctx512-20260928/`](delivery/norberto-lora-final-ctx512-20260928/): final training on 16,093 development rows.
- [`norberto-lora-final-holdout-20260928/`](delivery/norberto-lora-final-holdout-20260928/): separate evaluation on 3,999 held-out rows, with 45.31% accuracy.
- [`norberto-lora-test1-20260929/`](delivery/norberto-lora-test1-20260929/): predictions and audit for [`deliveries/test1.xlsx`](../../deliveries/test1.xlsx).

## Research, not promoted

[`experiments/`](experiments/) preserves the earlier NorBERTo 256/512 pilots and 256-token cross-validation. [`experiments/grid-search/`](experiments/grid-search/) holds the nine-configuration screening, the three-finalist CV, and the `candidate-08` final training/holdout evaluation. Its CV gain was marginal; the resulting adapter reached only 35.98% on the repeatedly used holdout and was **not** promoted. Do not treat this holdout as a fresh independent test or use it to tune further.

For the selection rationale and limitations, see the [model decision](../../docs/modelo-escolhido-norberto-512.md). Historical paths recorded inside completed runs are resolved by code when read; new runs use the canonical locations above.
