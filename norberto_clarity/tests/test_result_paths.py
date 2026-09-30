"""Historical result inputs must remain readable without filesystem aliases."""
from pathlib import Path

from clarity.result_paths import resolve_result_input


def test_resolves_historical_split_and_experiment_inputs():
    split = Path("results/dataset-hypotheses-20260923/splits.csv")
    experiment = Path("results/norberto-lora-development-ctx512-20260928")
    screening = Path("results/norberto-lora-grid-20260929-170333/summary.json")

    assert not split.exists()
    assert resolve_result_input(split) == Path(
        "results/research/dataset-hypotheses-20260923/splits.csv")
    assert resolve_result_input(experiment) == Path(
        "results/norberto/delivery/norberto-lora-development-ctx512-20260928")
    assert resolve_result_input(screening) == Path(
        "results/norberto/experiments/grid-search/"
        "norberto-lora-grid-20260929-170333/summary.json")


def test_preserves_canonical_and_new_output_paths():
    canonical = Path("results/research/dataset-hypotheses-20260923/splits.csv")
    new_output = Path("results/norberto/delivery/new-run")
    assert not any(path.is_symlink() for path in Path("results").iterdir())
    assert resolve_result_input(canonical) == canonical
    assert resolve_result_input(new_output) == new_output


def test_resolves_absolute_historical_input():
    from clarity.result_paths import REPOSITORY_ROOT

    old_path = REPOSITORY_ROOT / "results/dataset-hypotheses-20260923/splits.csv"
    assert resolve_result_input(old_path) == (
        REPOSITORY_ROOT / "results/research/dataset-hypotheses-20260923/splits.csv")


def test_historical_split_in_saved_manifest_remains_readable():
    from clarity.finetune_data import prepare_training_corpus

    frame, _, _, metadata = prepare_training_corpus(
        Path("data/train.xlsx"),
        Path("results/dataset-hypotheses-20260923/splits.csv"),
        "0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494",
    )
    assert len(frame) == 16093
    assert metadata["reserved_holdout_rows"] == 3999
    assert metadata["split_path"] == (
        "results/research/dataset-hypotheses-20260923/splits.csv")
