# Comparação lexical pareada no holdout fixo

## Protocolo pré-fixado

Esta rodada compara os dois modelos clássicos que permaneceram indistinguíveis na avaliação histórica: regressão logística lexical com `C=0,25` e LinearSVC lexical com `C=0,05`. Ambos usam o mesmo TF-IDF de palavras e caracteres, os mesmos 16.093 exemplos de desenvolvimento, os mesmos folds agrupados e as mesmas 3.999 linhas de validação fixa. Não há busca de novos hiperparâmetros.

A [configuração](../../configs/research/fixed-lexical-comparison.json) fixa o SHA-256 `0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494` para a [partição](../../results/research/dataset-hypotheses-20260923/splits.csv). O executor rejeita qualquer arquivo de split diferente, mesmo que preserve contagens e ausência de sobreposição entre grupos.

O vencedor para o artefato é escolhido somente pela acurácia média da validação cruzada no desenvolvimento; empate favorece a regressão logística, que aparece primeiro. Os dois modelos são posteriormente avaliados nas mesmas linhas do holdout. A comparação principal é `LinearSVC − regressão logística` em acurácia, acompanhada por intervalo descritivo obtido por bootstrap dos grupos textuais. F1 macro, acurácia balanceada, matrizes de confusão e predições linha a linha também serão preservadas.

Este conjunto já orientou uma decisão anterior e, portanto, é validação fixa reutilizada, não teste final independente. Um ganho pontual pequeno ou um intervalo que inclua zero não sustentará uma afirmação de superioridade geral.

## Execução

```bash
PYTHONPATH=src python -m model_research.holdout_train \
  --train data/train.xlsx \
  --split results/research/dataset-hypotheses-20260923/splits.csv \
  --config configs/research/fixed-lexical-comparison.json \
  --output artifacts/fixed-lexical-comparison-20260924.joblib \
  --folds 3 --seed 42 --threads 2
```

O destino é novo e não sobrescreve o artefato histórico.

## Resultado auditado

A execução terminou em 24/09/2026 sem erros ou avisos de convergência. As buscas, incluindo os reajustes no desenvolvimento completo, levaram 242,81 segundos. O [auditor independente](../../src/model_research/audit_holdout.py) conferiu o dataset, o SHA-256 do split, as linhas e grupos do holdout, os rótulos originais, as métricas, a comparação pareada e a reprodução das predições pelo modelo serializado. A [auditoria salva](../../results/research/fixed-lexical-comparison-20260924/audit.json) terminou com `status=passed`.

O bootstrap reamostrou os componentes de similaridade identificados por `group_id` na partição congelada. A metadata gerada durante a execução preserva o rótulo genérico `normalized_text_group` herdado do helper; a auditoria registra a designação operacional correta, `frozen_similarity_component`. Essa correção de nomenclatura não altera as amostras nem o intervalo calculado.

| Modelo | CV no desenvolvimento | Acurácia fixa | F1 macro | Acurácia balanceada | Acertos |
|---|---:|---:|---:|---:|---:|
| Regressão logística lexical, C=0,25 | 44,00% | **45,09%** | **44,78%** | **45,29%** | **1.803** |
| LinearSVC lexical, C=0,05 | **44,10%** | 44,81% | 44,47% | 45,04% | 1.792 |

O critério pré-fixado de desenvolvimento selecionou o LinearSVC por 0,095 p.p. Entretanto, nas mesmas 3.999 linhas da validação fixa, a regressão logística obteve 11 acertos adicionais, ganho de 0,275 p.p. O intervalo descritivo de 95% para `LinearSVC − logística` foi de **−0,780 a +0,202 p.p.**, incluindo zero. Os modelos discordaram em 147 linhas: ambos acertaram 1.748, somente a logística acertou 55, somente o SVM acertou 44 e ambos erraram 2.152.

A vantagem da logística concentrou-se na classe intermediária: recall de 35,00%, contra 34,02% do SVM. O SVM ficou 0,16 p.p. acima em recall de `c1` e 0,08 p.p. acima em `c5`, diferenças pequenas. As [metadatas completas](../../results/research/fixed-lexical-comparison-20260924/metadata.json) e as [predições pareadas](../../results/research/fixed-lexical-comparison-20260924/holdout-predictions.csv) foram preservadas.

## Decisão

O LinearSVC não demonstrou vantagem consistente: ganhou marginalmente no desenvolvimento e perdeu marginalmente na validação fixa, com intervalo pareado incluindo zero. Portanto, ele não será promovido como melhoria. A regressão logística lexical permanece a referência conservadora e o candidato clássico preferido, por combinar o melhor resultado fixo desta comparação com F1 macro e acurácia balanceada também superiores. Isso não prova superioridade geral nem substitui a avaliação oficial externa.

O artefato desta rodada contém o LinearSVC porque o executor obedeceu ao critério pré-fixado de seleção exclusivamente pela CV de desenvolvimento. Ele é evidência auditável do protocolo, não o modelo escolhido. A regressão logística foi [salva separadamente como referência lexical](modelo-final-logistico.md); a decisão posterior escolheu o [NorBERTo-512](../modelo-escolhido-norberto-512.md) para a entrega.
