# Guia da documentação

Comece pela [decisão do modelo escolhido](modelo-escolhido-norberto-512.md). O **NorBERTo-base com LoRA e contexto de 512 tokens** é o modelo de entrega; a regressão logística lexical permanece como referência, não como o modelo escolhido. O [desenvolvimento](desenvolvimento.md) preserva a cronologia técnica; os relatórios detalhados do modelo estão no [índice NorBERTo](norberto/README.md) e os artefatos de avaliação no [índice de resultados](../results/README.md).

## Modelo escolhido e avaliações

| Pergunta | Documento |
|---|---|
| Qual modelo foi escolhido, por quê e onde está salvo? | [Decisão e artefato NorBERTo-512](modelo-escolhido-norberto-512.md) |
| Como foi avaliado nos três folds de desenvolvimento e treinado novamente? | [Avaliação cruzada e treino final](norberto/avaliacao-norberto-512-e-treino-final.md) |
| Qual foi o resultado do modelo salvo no holdout fixo? | [Avaliação separada no holdout](norberto/avaliacao-norberto-holdout-final.md) |
| Qual é a referência lexical e como foi salva? | [Regressão logística de referência](research/modelo-final-logistico.md) |
| Como SVM e logística se comparam no mesmo holdout? | [Comparação lexical pareada](research/comparacao-lexical-holdout-fixo.md) |

Os **43,47% OOF** do NorBERTo-512 nos folds de desenvolvimento e os **45,31%** do modelo final no holdout fixo medem conjuntos e etapas diferentes. Não constituem uma comparação direta entre contextos ou uma prova de ganho sobre a logística. O holdout já foi consultado em pesquisas anteriores; a planilha `test1` foi preenchida, mas ainda não tem gabarito externo para avaliar acurácia.

O ponto de entrada `python src/main.py` aceita `train` para treinar novamente somente com o desenvolvimento e `test` para preencher uma planilha sem rótulos. Veja os comandos no [README principal](../README.md). A entrega versionada está em [`deliveries/test1.xlsx`](../deliveries/test1.xlsx).

## Experimentos e diagnóstico

- [Protocolo registrado para os três folds do NorBERTo-512](norberto/experimento-norberto-512-desenvolvimento.md).
- [Screening posterior de hiperparâmetros do NorBERTo-512](norberto/screening-hiperparametros.md), concluído nas nove configurações, com ranking e TOP 3 no desenvolvimento.
- [CV separada dos três finalistas](norberto/cv-finalistas-hiperparametros.md), concluída com `candidate-08` à frente por pequena margem, sem uso do holdout ou promoção automática ao modelo entregue.
- [Treino final exploratório do vencedor](norberto/treino-final-candidate-08.md), concluído no desenvolvimento e com adapter próprio; o modelo e a planilha de entrega anteriores permanecem vigentes.
- [Avaliação separada do novo adapter no holdout](norberto/avaliacao-holdout-candidate-08.md): resultado inferior ao adapter vigente, que foi mantido.
- [Pilotos BERTimbau e NorBERTo-256/512, com decisões históricas](research/experimento-bertimbau.md). Recomendações de contexto 256 ali descritas pertencem à etapa exploratória anterior, não substituem a decisão final pelo contexto 512.
- [MiniLM congelado com MLP](research/experimento-semantic-mlp.md) e [avaliação do LinearSVC](research/avaliacao-linear-svc.md).
- [Diagnóstico do dataset](research/analise-problemas-dataset.md), [limpeza](research/experimento-limpeza-dataset.md) e [hipóteses prospectivas](research/experimento-hipoteses-dataset.md). Nenhum tratamento foi promovido ao conjunto original.
- [Primeira avaliação SVM no holdout](research/avaliacao-svm-holdout-fixo.md) e [compatibilidade Docker/ROCm](research/compatibilidade-rocm.md).

Os [relatórios históricos de outros modelos e tratamentos](research/README.md) ficam separados da implementação final, mas continuam reproduzíveis.

## Enunciado e entrega

- Enunciado: [PDF](enunciado/ep1-enunciado.pdf) e [transcrição](enunciado/ep1-enunciado-transcricao.md). Modelo de relatório: [PDF](enunciado/ep1-modelo-relatorio.pdf) e [transcrição](enunciado/ep1-modelo-relatorio-transcricao.md).
- [Entrega de `data/test1.xlsx` com NorBERTo-512](norberto/entrega-test1-norberto-512.md): arquivo em `deliveries/test1.xlsx`, **comando Python direto**, requisitos CPU/NVIDIA/AMD, probabilidades, hashes e auditoria. A entrada pública usa `python src/main.py test`; a CLI clássica está em `PYTHONPATH=src python -m model_research.cli`. A planilha de teste não traz gabarito, portanto não há acurácia oficial para ela.

Relatórios de experimentos preservam as conclusões e os comandos válidos **à época**. Seus comandos Docker documentam o ambiente de treino local, não são requisitos para executar a entrega. Para inferência em outra máquina, use o comando Python direto do guia acima; para a decisão vigente, consulte a página do modelo escolhido e o registro de desenvolvimento.
