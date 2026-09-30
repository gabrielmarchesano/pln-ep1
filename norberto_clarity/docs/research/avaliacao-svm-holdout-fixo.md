# SVM lexical com holdout fixo — 23/09/2026

## Protocolo

Foi usada a planilha original `data/train.xlsx` (SHA-256 `0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`) e a [partição congelada](../../results/research/dataset-hypotheses-20260923/splits.csv) (SHA-256 `0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`). Os 16.093 exemplos de desenvolvimento alimentaram a validação cruzada agrupada em três folds e o ajuste final; os 3.999 exemplos de holdout foram usados apenas para predição. Os grupos de textos similares da partição não cruzam os dois lados. Nenhuma linha ou rótulo da planilha foi modificado.

O candidato foi `LinearSVC` com TF-IDF de palavras e caracteres e `C=0,05`, [pré-fixado na configuração](../../configs/research/final-svm.json) com base nas rodadas anteriores. Não houve busca de novos valores neste holdout. O treino foi executado em contêiner ROCm, mas TF-IDF e `LinearSVC` usam CPU. O tempo de busca e ajuste registrado foi 74,44 segundos, incluindo 19,35 segundos do reajuste no desenvolvimento completo.

## Resultados

| Conjunto | Acurácia | F1 macro | Observação |
|---|---:|---:|---|
| Validação cruzada no desenvolvimento | 44,10% | — | Média dos folds: 44,83%, 44,13%, 43,32%; o executor de busca registra apenas acurácia interna. |
| Holdout fixo | **44,81%** | **44,47%** | 1.792 acertos em 3.999 linhas; acurácia balanceada 45,04%. |

No holdout, os recalls foram 43,75% para `c1`, 34,02% para `c234` e 57,36% para `c5`. A classe intermediária continua sendo a mais difícil. Matriz de confusão, com linhas verdadeiras e colunas previstas na ordem `c1`, `c234`, `c5`:

| Rótulo | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 550 | 331 | 376 |
| c234 | 385 | 482 | 550 |
| c5 | 245 | 320 | 760 |

A [metadata completa](../../results/research/svm-fixed-holdout-20260923/metadata.json) e as [3.999 predições](../../results/research/svm-fixed-holdout-20260923/holdout-predictions.csv) foram preservadas. A auditoria recalculou acurácia, F1 e matriz a partir das predições, conferiu os rótulos com as linhas originais e reproduziu as previsões com o artefato salvo em `artifacts/svm-fixed-holdout-20260923.joblib`. A suíte completa passou: 57 testes aprovados, três pulados.

## Interpretação e uso futuro

Os 44,81% do holdout não demonstram melhoria sobre os 45,18% históricos: as partições e populações de avaliação são diferentes. A proximidade entre 44,10% da CV e 44,81% do holdout é compatível com variação de amostragem; não a interpretar como ganho de generalização. O modelo continua em avaliação e não substitui automaticamente um artefato de produção.

Por decisão do projeto, toda nova rodada deve manter esta partição fixa: seleção de parâmetros e ajuste **somente nos 16.093 exemplos de desenvolvimento**, seguidos de avaliação dos mesmos 3.999 exemplos de holdout. Comparações futuras devem usar as mesmas linhas e métricas pareadas. Como as pontuações desse holdout vão orientar mudanças, ele deixa de ser um teste final independente e passa a ser **validação fixa reutilizada**. Além disso, as rodadas históricas examinaram a base inteira antes da criação da partição; uma estimativa final não adaptativa exigirá outro conjunto externo, como o teste oficial ainda indisponível.

Para repetir com um novo destino, use:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace -e PYTHONPATH=/workspace/src pln-rocm:6.4.1 \
  python3 -m model_research.holdout_train \
  --train data/train.xlsx \
  --split results/research/dataset-hypotheses-20260923/splits.csv \
  --config configs/research/final-svm.json \
  --output artifacts/svm-fixed-holdout-new-run.joblib \
  --folds 3 --seed 42 --threads 2
```

O executor rejeita destinos já existentes, compara o SHA-256 da planilha com o manifesto e exige que o SHA-256 da partição seja exatamente o fixado na configuração antes do treinamento.

Uma [comparação pareada posterior](comparacao-lexical-holdout-fixo.md) avaliou também a regressão logística `C=0,25` nas mesmas 3.999 linhas. Ela obteve 45,09% contra 44,81% do SVM, diferença pequena cujo intervalo descritivo inclui zero. Assim, o SVM não foi promovido como melhoria.
