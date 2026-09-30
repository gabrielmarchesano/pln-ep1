# NorBERTo-512 hyperparameter screening

Nine configurations were evaluated on the same fixed grouped development split. No three-fold CV or final holdout evaluation was run.

Accuracy differences below are absolute proportions; 0.01 means one percentage point. The baseline is candidate-03 (learning_rate=1e-3, rank=4, alpha=8).

| Rank | Candidate | LR | LoRA rank | Alpha | Accuracy | Macro-F1 | Δ baseline | Gap to #1 | Gap to #3 | Runtime (min) | Best epoch | Best checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | candidate-05 **TOP 3** | 3e-04 | 8 | 16 | 0.4455 | 0.4461 | +0.0624 | 0.0000 | -0.0074 | 45.3 | 2.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-05/checkpoints/checkpoint-1624` |
| 2 | candidate-08 **TOP 3** | 3e-04 | 16 | 32 | 0.4429 | 0.4429 | +0.0598 | 0.0026 | -0.0048 | 46.8 | 2.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-08/checkpoints/checkpoint-1624` |
| 3 | candidate-02 **TOP 3** | 3e-04 | 4 | 8 | 0.4381 | 0.4373 | +0.0550 | 0.0074 | +0.0000 | 45.3 | 3.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-02/checkpoints/checkpoint-2436` |
| 4 | candidate-04  | 1e-04 | 8 | 16 | 0.4288 | 0.4284 | +0.0456 | 0.0167 | +0.0093 | 45.3 | 3.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-04/checkpoints/checkpoint-2436` |
| 5 | candidate-07  | 1e-04 | 16 | 32 | 0.4275 | 0.4275 | +0.0444 | 0.0180 | +0.0106 | 46.0 | 3.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-07/checkpoints/checkpoint-2436` |
| 6 | candidate-01  | 1e-04 | 4 | 8 | 0.4195 | 0.4190 | +0.0363 | 0.0260 | +0.0186 | 45.6 | 3.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-01/checkpoints/checkpoint-2436` |
| 7 | candidate-06  | 1e-03 | 8 | 16 | 0.3941 | 0.3445 | +0.0109 | 0.0514 | +0.0440 | 45.3 | 2.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-06/checkpoints/checkpoint-1624` |
| 8 | candidate-03  | 1e-03 | 4 | 8 | 0.3832 | 0.3009 | +0.0000 | 0.0624 | +0.0550 | 30.8 | 1.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-03/checkpoints/checkpoint-812` |
| 9 | candidate-09  | 1e-03 | 16 | 32 | 0.3562 | 0.1751 | -0.0270 | 0.0894 | +0.0820 | 31.9 | 1.0 | `artifacts/norberto-lora-grid-20260929-170333/screening/candidate-09/checkpoints/checkpoint-812` |

## Decision gaps

- #1 vs #2: 0.0026
- #2 vs #3: 0.0048
- #3 vs #4: 0.0093
- #1 vs #4: 0.0167

These single-split results are exploratory. Review the size and stability of the gaps before deciding which candidates warrant three-fold CV.
