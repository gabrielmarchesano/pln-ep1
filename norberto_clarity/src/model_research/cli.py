from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from clarity.data import LABELS, LABEL_COLUMN, TEXT_COLUMN, describe_dataset, file_digest
from clarity.data import load_training_data, read_frame, text_groups
from clarity.delivery import require_new_file, write_predictions
from .experiment import environment, evaluate, read_config, run_experiment
from .experiment import search_candidates, write_json
from .models import MODEL_NAMES, build_model


def train_and_predict(train_path, test_path, output_path, model_name):
    train = load_training_data(train_path)
    test = read_frame(test_path, training=False)
    model = build_model(model_name).fit(train[TEXT_COLUMN], train[LABEL_COLUMN])
    write_predictions(test_path, output_path, model.predict(test[TEXT_COLUMN]))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="EP1: classificação de clareza com validação agrupada.")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Auditar treino e duplicatas.")
    inspect.add_argument("--train", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    embed = commands.add_parser("embed", help="Preparar o cache semântico sem usar rótulos.")
    embed.add_argument("--train", type=Path, required=True)
    embed.add_argument("--backend", choices=["static", "transformer"], default="static")
    embed.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    embed.add_argument("--batch-size", type=int, default=16)
    for name in ("evaluate", "experiment", "train"):
        command = commands.add_parser(name)
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--folds", type=int, default=3)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--threads", type=int, default=2)
        if name == "evaluate":
            command.add_argument("--model", choices=MODEL_NAMES, default="baseline")
            command.add_argument("--output", type=Path, default=Path("results.json"))
        else:
            command.add_argument("--config", type=Path, default=Path("configs/research/classical.json"))
            command.add_argument("--cache-dir", default=".cache/sklearn")
            command.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0,
                                 help="Progresso da busca: 0 resumido, 1 contagem, 2 cada ajuste.")
            if name == "experiment":
                command.add_argument("--inner-folds", type=int, default=3)
                command.add_argument("--output-dir", type=Path, required=True)
            else:
                command.add_argument("--output", type=Path, required=True,
                                     help="Modelo .joblib e metadados .json adjacentes.")
    predict = commands.add_parser("predict")
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path, help="Modelo .joblib deste projeto; use apenas arquivos confiáveis.")
    source.add_argument("--train", type=Path, help="Compatibilidade: treinar modelo fixo antes de prever.")
    predict.add_argument("--model", choices=MODEL_NAMES, default="baseline")
    predict.add_argument("--test", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--threads", type=int, default=2)
    return root


def execute(args):
    if args.command == "embed":
        from .semantic import FrozenEmbeddings
        from .static_embeddings import StaticEmbeddings
        frame = read_frame(args.train, training=False)
        encoder = StaticEmbeddings() if args.backend == "static" else FrozenEmbeddings(
            device=args.device, batch_size=args.batch_size)
        vectors = encoder.fit_transform(frame[TEXT_COLUMN])
        print(f"Encoder {args.backend} pronto: {vectors.shape}")
        return
    if args.command == "predict":
        require_new_file(args.output)
        test = read_frame(args.test, training=False)
        if args.artifact:
            bundle = joblib.load(args.artifact)
            if bundle.get("format_version") != 1 or tuple(bundle.get("labels", ())) != LABELS:
                raise ValueError("Artefato incompatível com o formato deste projeto.")
            current = environment()["packages"].get("scikit-learn")
            saved = bundle["metadata"]["environment"]["packages"].get("scikit-learn")
            if saved != current:
                raise ValueError(f"Use a versão de scikit-learn do treino: {saved} (atual: {current}).")
            write_predictions(args.test, args.output, bundle["model"].predict(test[TEXT_COLUMN]))
        else:
            if args.train.resolve() == args.output.resolve():
                raise ValueError("O destino não pode sobrescrever o treino.")
            train_and_predict(args.train, args.test, args.output, args.model)
        print(f"Planilha conferida e salva: {args.output}")
        return
    frame = load_training_data(args.train)
    if args.command == "inspect":
        result = {"dataset_sha256": file_digest(args.train), "dataset": describe_dataset(frame)}
    elif args.command == "evaluate":
        require_new_file(args.output)
        result = {"dataset_sha256": file_digest(args.train), "environment": environment(),
                  "dataset": describe_dataset(frame),
                  "evaluation": evaluate(frame, args.model, args.folds, args.seed)}
    elif args.command == "experiment":
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("O diretório do experimento deve ser novo ou vazio.")
        result = run_experiment(
            frame, args.train, read_config(args.config), args.output_dir,
            folds=args.folds, inner_folds=args.inner_folds, seed=args.seed,
            cache_dir=args.cache_dir, verbose=args.verbose,
        )
        for name, summary in result["models"].items():
            print(f"{name}: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"Resultados e predições fora do treino: {args.output_dir}")
        return
    else:
        if args.output.suffix != ".joblib":
            raise ValueError("O modelo deve ser salvo como .joblib.")
        require_new_file(args.output)
        metadata_path = args.output.with_suffix(".json")
        require_new_file(metadata_path)
        config = read_config(args.config)
        x, y = frame[TEXT_COLUMN].to_numpy(), frame[LABEL_COLUMN].to_numpy()
        fitted, searches, winner = search_candidates(
            x, y, text_groups(x), config, folds=args.folds, seed=args.seed,
            cache_dir=args.cache_dir, verbose=args.verbose,
        )
        model = fitted[winner]
        model.memory = None
        result = {"selected": winner, "searches": searches, "config": config,
                  "dataset_sha256": file_digest(args.train), "dataset": describe_dataset(frame),
                  "folds": args.folds, "seed": args.seed, "environment": environment(),
                  "note": "Seleção para treino final. Scores internos não são estimativa de teste."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format_version": 1, "labels": LABELS, "model": model,
                     "metadata": result}, args.output, compress=3)
        write_json(metadata_path, result)
        print(f"Modelo selecionado: {winner}; salvo em {args.output}")
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        require_new_file(args.output)
        write_json(args.output, result)


def main():
    cli = parser()
    args = cli.parse_args()
    if getattr(args, "threads", 2) < 1:
        cli.error("--threads deve ser positivo.")
    try:
        with threadpool_limits(limits=getattr(args, "threads", 2)):
            execute(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        cli.exit(2, f"Erro: {error}\n")


if __name__ == "__main__":
    main()
