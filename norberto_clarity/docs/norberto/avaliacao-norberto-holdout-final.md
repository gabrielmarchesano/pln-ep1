# Avaliação do NorBERTo-512 final no holdout fixo

Esta é a avaliação posterior do [NorBERTo-512 escolhido](../modelo-escolhido-norberto-512.md), não um novo experimento de seleção nem um novo treinamento.

O comando separado `evaluate-holdout` concluiu a inferência em **3.999 respostas** do holdout congelado. Carregou exclusivamente o adapter salvo em `artifacts/norberto-lora-final-ctx512-20260928/final_model/`, sem novo treinamento. Os [resultados completos](../../results/norberto/delivery/norberto-lora-final-holdout-20260928/results.json), as [predições e probabilidades](../../results/norberto/delivery/norberto-lora-final-holdout-20260928/holdout-predictions.csv) e a [auditoria](../../results/norberto/delivery/norberto-lora-final-holdout-20260928/audit.json) estão preservados.

| Modelo avaliado nas mesmas 3.999 linhas | Acurácia | F1 macro | Acurácia balanceada | Acertos |
|---|---:|---:|---:|---:|
| NorBERTo-512 final, LoRA | **45,31%** | **45,31%** | 45,31% | 1.812 |
| Regressão logística lexical, rodada fixa anterior | 45,09% | 44,78% | 45,29% | 1.803 |

O NorBERTo acertou **nove linhas a mais**, diferença de **+0,23 ponto percentual**. Houve 516 acertos exclusivos do NorBERTo e 507 exclusivos da logística. O intervalo descritivo pareado de 95%, reamostrando os componentes de similaridade congelados, foi **[−1,51; +2,08] p.p.** Inclui zero: o pequeno ganho observado não estabelece superioridade geral. A logística usada nessa comparação é a que foi treinada apenas no desenvolvimento e avaliada no [mesmo holdout](../research/comparacao-lexical-holdout-fixo.md); seu artefato posteriormente reajustado nas 20.092 linhas não seria uma referência válida nessas linhas, pois já as viu no treino.

| Classe | Suporte | Recall NorBERTo | Recall logística | F1 NorBERTo |
|---|---:|---:|---:|---:|
| `c1` | 1.257 | 41,85% | 43,60% | 44,09% |
| `c234` | 1.417 | **42,70%** | 35,00% | **42,17%** |
| `c5` | 1.325 | 51,40% | 57,28% | 49,65% |

O ganho mais relevante para diagnóstico está em `c234`, com **+7,70 p.p. de recall** em relação à logística; parte da troca é a redução do recall de `c5` em 5,89 p.p. Esse resultado mostra uma redistribuição dos erros, não precisão uniformemente maior em todas as classes.

Matriz de confusão do NorBERTo, linhas verdadeiras e colunas previstas na ordem `c1`, `c234`, `c5`:

| Classe verdadeira | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 526 | 436 | 295 |
| c234 | 370 | 605 | 442 |
| c5 | 233 | 411 | 681 |

A auditoria conferiu todas as linhas, rótulos e grupos contra a planilha e a avaliação lexical, recalculou as métricas e a matriz, verificou soma e finitude das probabilidades, reconfirmou o hash do adapter e registrou `training_performed=false`. O modelo havia sido treinado por **duas épocas**, escolhidas pela mediana das melhores épocas internas dos três folds (2, 3, 2), em **16.093 exemplos de desenvolvimento**, com sobreposição zero com o holdout. O [relatório de treino e CV](avaliacao-norberto-512-e-treino-final.md) documenta a seleção, os controles e os artefatos.

Na CV de desenvolvimento, NorBERTo ficou em 43,47% contra 43,99% da logística lexical; no holdout fixo, a ordem pontual se inverteu. Isso reforça a incerteza da comparação. A escolha do NorBERTo como candidato de entrega pode ser defendida pela arquitetura contextual pré-treinada e pela adaptação LoRA, importantes para o critério de inovação, **sem alegar ganho preditivo comprovado**. O holdout já havia sido consultado em decisões lexicais e toda a base fora examinada em rodadas históricas; esta é a avaliação final do **modelo salvo**, mas não uma estimativa historicamente independente. A [planilha `test1` foi preenchida](entrega-test1-norberto-512.md), mas não traz rótulos verdadeiros para uma avaliação quantitativa externa.
