# Pesquisa histórica de modelos

Este pacote contém os experimentos clássicos, semânticos e de tratamento do dataset. A entrada pública `src/main.py` não o importa. A única dependência intencional entre pacotes é que os modos históricos `pilot` e `experiment` do NorBERTo carregam controles lexicais de `model_research.models`; `final`, `evaluate-holdout` e `predict-test` não treinam nem carregam esses controles.

Execute comandos históricos na raiz do repositório com `PYTHONPATH=src`, por exemplo `python -m model_research.cli --help` ou `python -m model_research.dataset.cleaning --help`. Use diretórios de saída novos; os resultados versionados são evidências imutáveis. A documentação histórica está indexada em `docs/research/README.md`.
