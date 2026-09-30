# Avaliação cruzada dos três finalistas do screening

O [screening de nove configurações](screening-hiperparametros.md) selecionou `candidate-05`, `candidate-08` e `candidate-02`, nessa ordem. Esta rodada avalia **somente** esses três candidatos, cada um com o modo `experiment` já existente, nos mesmos três folds externos agrupados. Cada fold mantém a validação interna agrupada para escolha do checkpoint, a seed 42, o contexto 512 e os demais parâmetros aprovados. O runner `clarity.finetune_grid_cv` reaproveita `clarity.finetune.run()`; não reimplementa a divisão ou o treinamento dos folds.

Antes de criar saídas, o runner exige screening completo, nove resultados coerentes, TOP 3 derivado do ranking, configurações e hashes intactos, dataset original e split de holdout congelado. `prepare_training_corpus()` exclui as 3.999 linhas de holdout; IDs de linhas e grupos são conferidos novamente. O modo `--preflight` executa essas verificações sem treinar nem criar arquivos. Nenhum treino final nem avaliação de holdout ocorre nesta rodada.

Os diretórios novos `results/norberto/experiments/grid-search/norberto-lora-grid-cv-<id>/` e `artifacts/norberto-lora-grid-cv-<id>/` separam a CV do screening e do adapter de entrega. O primeiro contém `summary.json`, `cv_results.csv`, `run.log`, `pid.txt` e `candidates/candidate-XX/` com manifests, folds, predições OOF, métricas e controles lexicais do protocolo original. O segundo guarda os adapters e checkpoints por candidato e fold. O resumo é atualizado ao fim de cada candidato; após os três, ordena a **acurácia média dos folds** e desempata por F1 macro médio, rank menor, taxa menor e ID. Uma falha ou parada por orçamento impede declarar a CV completa.

Para conferir os arquivos durante a execução, use `tail -f results/norberto/experiments/grid-search/norberto-lora-grid-cv-<id>/run.log` e leia `summary.json`. Um exemplo de verificação prévia, sem saída nova, é:

```bash
PYTHONPATH=src python -m clarity.finetune_grid_cv --preflight \
  --screening-dir results/norberto/experiments/grid-search/norberto-lora-grid-20260929-170333 \
  --output-dir results/norberto/experiments/grid-search/norberto-lora-grid-cv-<id> \
  --artifact-dir artifacts/norberto-lora-grid-cv-<id>
```

Os scores de CV são **exploratórios**: os candidatos foram selecionados após observar o screening sobre o mesmo conjunto de desenvolvimento. Não interpretar a CV como teste independente nem trocar automaticamente o adapter final ou a planilha já entregue. A decisão sobre eventual novo treinamento final fica para depois de analisar os resultados completos.

## Resultado e decisão — 30/09/2026

A [CV dos três candidatos](../../results/norberto/experiments/grid-search/norberto-lora-grid-cv-20260929-233539/summary.json) terminou sem parada por orçamento. A [auditoria independente](../../results/norberto/experiments/grid-search/norberto-lora-grid-cv-20260929-233539/candidates/candidate-08/audit.json) do vencedor confirmou 16.093 predições OOF, três folds, splits e grupos válidos, probabilidades e métricas recalculadas, além do isolamento das 3.999 linhas de holdout. As auditorias dos outros dois candidatos também passaram. Os arquivos de fold, as predições e os adapters permanecem separados por candidata.

| Posição CV | Candidata | Taxa; rank/alpha | Acurácia média ± desvio | F1 macro médio | Acurácia nos folds 1/2/3 | Tempo total |
|---:|---|---|---:|---:|---|---:|
| 1 | `candidate-08` | 0,0003; 16/32 | **43,73% ± 0,91 p.p.** | 42,81% | 44,93% / 43,53% / 42,74% | 98,3 min |
| 2 | `candidate-02` | 0,0003; 4/8 | 43,17% ± 0,74 p.p. | 42,32% | 44,19% / 42,46% / 42,86% | 104,8 min |
| 3 | `candidate-05` | 0,0003; 8/16 | 41,23% ± 2,99 p.p. | 39,22% | 43,93% / 42,70% / 37,06% | 98,7 min |

O líder do screening, `candidate-05`, caiu para terceiro; seu fold 3 teve forte deterioração. `candidate-08` venceu pela maior acurácia **média** dos três folds, não em todos os folds individualmente. A configuração anterior (`learning_rate=0,001`, rank 4, alpha 8) havia obtido **43,49% de acurácia média e 42,89% de F1 macro médio** nos mesmos folds. A nova candidata ganha apenas **0,25 p.p. de acurácia OOF agregada**, enquanto o F1 macro OOF cai aproximadamente **0,13 p.p.** As 16.093 linhas e os IDs de fold foram conferidos como idênticos antes da comparação pareada. O intervalo descritivo de 95% por reamostragem de grupos para `candidate-08` menos a configuração anterior é **[−0,37; +0,84] p.p.**; inclui zero. Frente a `candidate-02`, a diferença OOF é **+0,54 p.p.**, intervalo **[−0,22; +1,24] p.p.** Também não há evidência de melhoria estável nessa comparação.

No relatório de classes OOF, `candidate-08` elevou o recall de `c5` de 50,87% para 55,74%, mas reduziu o de `c234` de 33,43% para 30,22% em relação à configuração anterior. A melhora pontual de acurácia não deve ser descrita como superioridade geral ou como ganho uniforme nas classes. A referência lexical completa ficou em 44,00% de acurácia média nos mesmos folds; `candidate-08` ficou 0,27 p.p. abaixo. A busca e esta CV usaram repetidamente o desenvolvimento, portanto os intervalos são apenas descritivos e não removem o viés de seleção adaptativa.

**Decisão operacional:** `candidate-08` é o campeão pela métrica primária predefinida (acurácia média dos folds). Um **novo** NorBERTo-base com LoRA rank 16/alpha 32 e taxa 0,0003 foi [treinado nas 16.093 linhas de desenvolvimento](treino-final-candidate-08.md), sem reaproveitar o adapter de um fold. As melhores épocas foram 2, 3 e 3; a regra já implementada escolheu a **mediana, 3 épocas**, sem validação ou early stopping nesse treino. O holdout não foi avaliado automaticamente. Até avaliar o novo adapter separadamente e decidir uma promoção, o modelo e a planilha de entrega existentes permanecem inalterados.
