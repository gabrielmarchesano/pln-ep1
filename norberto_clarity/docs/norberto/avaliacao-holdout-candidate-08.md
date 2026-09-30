# Avaliação separada do `candidate-08` no holdout fixo

O adapter LoRA [treinado após a busca de hiperparâmetros](treino-final-candidate-08.md) foi avaliado em **3.999 linhas do holdout congelado**, somente por inferência, sem novo treino nem ajuste de configuração. O [resultado](../../results/norberto/experiments/grid-search/norberto-lora-grid-final-holdout-20260930-141916/results.json), as [predições linha a linha](../../results/norberto/experiments/grid-search/norberto-lora-grid-final-holdout-20260930-141916/holdout-predictions.csv) e a [auditoria](../../results/norberto/experiments/grid-search/norberto-lora-grid-final-holdout-20260930-141916/audit.json) foram preservados em diretório novo. Os hashes do dataset, split e adapter conferiram com o treino final.

| Modelo salvo | Acurácia | F1 macro | Acurácia balanceada | Acertos / 3.999 |
|---|---:|---:|---:|---:|
| `candidate-08`, rank 16, taxa 0,0003, 3 épocas | **35,98%** | **35,31%** | 35,91% | 1.439 |
| Adapter anterior, rank 4, taxa 0,001, 2 épocas | **45,31%** | **45,31%** | 45,31% | 1.812 |

As predições foram pareadas nos **mesmos IDs de linha, grupos e rótulos**. O novo adapter perdeu **373 acertos**, diferença de **−9,33 pontos percentuais**. A reamostragem descritiva de 2.000 conjuntos de componentes de similaridade forneceu intervalo de 95% **[−11,80; −6,94] p.p.** para `candidate-08` menos o anterior. Ele inclui incerteza condicionada às predições salvas, não todo o processo de seleção e treinamento. Entre as respostas, 644 foram acertadas apenas pelo novo modelo, 1.017 apenas pelo anterior, 795 por ambos e 1.543 por nenhum.

| Classe | Recall novo | Recall anterior | Diferença |
|---|---:|---:|---:|
| `c1` | 26,41% | 41,85% | −15,43 p.p. |
| `c234` | 32,04% | 42,70% | −10,66 p.p. |
| `c5` | 49,28% | 51,40% | −2,11 p.p. |

A auditoria recalculou accuracy, F1 macro, acurácia balanceada e matriz de confusão, verificou probabilidades finitas e normalizadas, classes pelo argmax e alinhamento com a partição congelada. Como controle de serialização, o adapter salvo do fold 1 de `candidate-08` foi recarregado e reproduziu **exatamente as 5.322 predições** registradas na CV, com 44,93% de acurácia. Isso não demonstra por que o novo treino final deteriorou, mas afasta uma falha geral de recarga do adapter de rank 16. O volume de treino, a duração, o regime de validação e a seed diferem entre folds e treino final; isolá-los causalmente exigiria novos experimentos pré-planejados, sem escolher variantes pelo resultado deste holdout.

**Decisão:** não promover `candidate-08` à entrega. O adapter anterior em `artifacts/norberto-lora-final-ctx512-20260928/final_model/` e a planilha `deliveries/test1.xlsx` permanecem vigentes e não foram sobrescritos. O novo adapter fica preservado para pesquisa em `artifacts/norberto-lora-grid-final-20260930-130937/final_model/`. O holdout já havia sido consultado em pesquisas anteriores e nesta rodada; portanto, seus resultados são uma comparação exploratória, não uma estimativa independente de generalização.
