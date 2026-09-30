# Pesquisa e modelos alternativos

Esta área preserva a documentação dos modelos e hipóteses avaliados antes da escolha do NorBERTo-512. Não contém o procedimento de entrega vigente. A implementação correspondente está em [`src/model_research/`](../../src/model_research/), as configurações em [`configs/research/`](../../configs/research/) e os testes em [`tests/research/`](../../tests/research/). Os resultados físicos estão em [`results/research/`](../../results/research/README.md); o leitor de caminhos históricos mantém os manifests e auditorias utilizáveis sem links simbólicos ou alteração de hashes.

Para reproduzir a partir da raiz do repositório, instale em ambientes separados: `python -m pip install -r requirements/common.txt` para modelos clássicos; `requirements/research/semantic.txt` para MiniLM; `requirements/research/static.txt` para Model2Vec. Os experimentos supervisionados BERTimbau/NorBERTo usam o `requirements.txt` principal. Todos os pins foram preservados.

| Linha de pesquisa | Documentação |
|---|---|
| SVM, regressão logística e avaliação lexical | [LinearSVC](avaliacao-linear-svc.md), [SVM no holdout](avaliacao-svm-holdout-fixo.md), [comparação pareada](comparacao-lexical-holdout-fixo.md), [referência logística](modelo-final-logistico.md) |
| Embeddings e modelos semânticos | [MiniLM + MLP](experimento-semantic-mlp.md) |
| BERTimbau e pilotos anteriores do NorBERTo | [Histórico dos pilotos](experimento-bertimbau.md) |
| Diagnóstico e tratamentos do dataset | [Problemas](analise-problemas-dataset.md), [limpeza](experimento-limpeza-dataset.md), [hipóteses](experimento-hipoteses-dataset.md) |
| Ambiente usado nas execuções antigas | [Compatibilidade ROCm](compatibilidade-rocm.md) |

O [README anterior completo](README-historico.md) também foi preservado como registro de época. Comandos e caminhos nele refletem o fluxo histórico; para usar o modelo escolhido hoje, consulte o [README principal](../../README.md) e o [índice ativo](../README.md). Os resultados históricos não autorizam substituir o adapter final nem modificar `data/train.xlsx`.
