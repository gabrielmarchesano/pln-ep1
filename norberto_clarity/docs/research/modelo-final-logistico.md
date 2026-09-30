# Referência lexical salva — regressão logística

Este artefato foi chamado de “final” **na etapa lexical de 28/09/2026**. A decisão posterior escolheu o [NorBERTo-512 LoRA](../modelo-escolhido-norberto-512.md) como candidato de entrega. A regressão logística continua salva e auditada como **referência**, não como o modelo escolhido.

## Decisão lexical histórica e protocolo

A [comparação pareada](comparacao-lexical-holdout-fixo.md) manteve a regressão logística de palavras e caracteres com `C=0,25` como referência conservadora: 45,09% de acurácia e 44,78% de F1 macro na validação fixa, contra 44,81% e 44,47% do LinearSVC. A diferença é incerta e não demonstra superioridade geral, mas não sustenta promover o SVM.

Com a família e o hiperparâmetro encerrados naquela etapa, o artefato lexical foi reajustado nas 20.092 linhas de `data/train.xlsx`, sem modificar textos ou rótulos. A [configuração lexical](../../configs/research/final-logistic.json) contém um único candidato e um único valor de `C`; portanto, não existe nova escolha adaptativa. A CV agrupada de três folds foi registrada somente como diagnóstico interno antes do reajuste automático em toda a base.

O holdout fixo não foi consultado novamente **durante esse reajuste lexical**. Depois do reajuste completo, não resta uma avaliação interna intocada para esse artefato treinado em toda a base: seus resultados de referência permanecem os da comparação anterior, realizada com uma logística treinada somente no desenvolvimento. Posteriormente, o holdout foi usado para avaliar separadamente o NorBERTo escolhido; a estimativa independente de generalização depende do teste oficial externo.

## Execução

```bash
PYTHONPATH=src python -m model_research.cli train \
  --train data/train.xlsx \
  --config configs/research/final-logistic.json \
  --folds 3 --seed 42 --threads 2 \
  --output artifacts/final-logistic-20260928.joblib
```

O destino é novo e o executor recusa sobrescrita.

## Resultado e auditoria

A execução terminou em 28/09/2026. A acurácia diagnóstica média da CV foi **45,14%**, com desvio de 0,55 p.p. e folds de 45,91%, 44,86% e 44,65%. A busca de configuração única e seu reajuste levaram 43,29 segundos; o reajuste na base completa levou 9,72 segundos. A regressão convergiu em 49 iterações, sem avisos.

O artefato usa vocabulários de 60.000 atributos de palavras e 50.000 de caracteres. O [auditor](../../src/model_research/audit_artifact.py) confirmou formato, metadata, SHA-256 do dataset, configuração fixa, classes, parâmetros finitos, convergência, vocabulários, probabilidades normalizadas e predições nas 20.092 linhas. A acurácia aparente de treino, 64,88%, foi calculada apenas como teste de sanidade de serialização e **não** é estimativa de generalização.

- Artefato local: `artifacts/final-logistic-20260928.joblib`.
- SHA-256 do artefato: `f7af61142f43e1c8a37f0cd5a8191dd6d83ba1222cd8ecf12281f9929ac827c9`.
- [Metadata versionada](../../results/research/final-logistic-20260928/metadata.json).
- [Auditoria versionada](../../results/research/final-logistic-20260928/audit.json), com `status=passed`.

O artefato lexical está pronto para inferência pelo comando clássico `predict`, que preserva ordem, abas e colunas e reabre a saída para conferência. Esse comando **não carrega o adapter NorBERTo escolhido**. A entrega de [`test1.xlsx` com NorBERTo](../norberto/entrega-test1-norberto-512.md) usa o modo separado `predict-test`, que reutiliza o mesmo exportador de Excel. Não há gabarito de `test1` para calcular acurácia externa.
