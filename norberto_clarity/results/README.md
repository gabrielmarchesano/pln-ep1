# Results index

The physical results are organized by purpose:

- [`norberto/`](norberto/README.md): selected NorBERTo-512 delivery and subsequent experiments.
- [`research/`](research/README.md): dataset, lexical, semantic, and BERTimbau research.

The root contains only this index and the two result categories; no compatibility symlinks are required. Existing manifests and audit records retain their original path strings and hashes. The input path resolver in `src/clarity/result_paths.py` maps those historical paths to the physical files when a saved result is read. New runs should write to a new directory under the appropriate category. The frozen split's canonical path is `results/research/dataset-hypotheses-20260923/splits.csv`.

Model weights are stored separately under `artifacts/`. The selected LoRA adapter and verification metadata are tracked by Git; experimental adapters and the base-model weights are not. The selected adapter is identified in the [NorBERTo index](norberto/README.md).
