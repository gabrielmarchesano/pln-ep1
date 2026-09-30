# Screening de hiperparâmetros NorBERTo-512

Esta busca é **exploratória e posterior** à escolha do NorBERTo-512 como família do modelo de entrega. O adapter final existente e a planilha entregue permanecem intactos. Esta rodada executa **somente o screening das nove configurações** e para antes de qualquer CV de três folds. Não treina um novo modelo final nem avalia o holdout. O ranking serve para decidir quais candidatos, se houver, justificam uma avaliação mais cara depois.

## Dados e configuração congelados

- Fonte: `data/train.xlsx`, com 20.092 linhas originais. Somente as 16.093 linhas marcadas como desenvolvimento participam de treino, validação e ranking.
- Holdout: 3.999 linhas excluídas por `prepare_training_corpus()`, com verificação adicional de IDs de linha e `group_id`. SHA-256 da partição: `0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`.
- Base: `configs/norberto-lora-development-ctx512.json`, NorBERTo-base na revisão `db73446f89c96044863ea05a39f680524b84bccb`, contexto 512, truncamento `head_tail`, seed 42, três épocas, paciência 1, batch 4, acumulação 4 e demais parâmetros originais. O runner recusa alterações na configuração-base além dos três campos pesquisados.

## Protocolo de screening fixo

O único split é `grouped_splits(development, inner_folds=5, seed=42)[0]`. Seus IDs de linha e grupos são salvos em `screening-split.csv`, junto do hash do arquivo. As nove candidatas usam exatamente esse mesmo split e a mesma seed de treino. A grade é o produto ordenado dos ranks LoRA `{4, 8, 16}` com as taxas `{0,0001; 0,0003; 0,001}`, mantendo `lora_alpha = 2 × lora_rank`. Dropout, épocas e os demais hiperparâmetros não variam.

Cada treino reutiliza `train_one()` e salva o adapter selecionado pela validação interna. O runner recarrega o melhor checkpoint por meio do `Trainer`, calcula accuracy, F1 macro e balanced accuracy na validação fixa e registra época, checkpoint, perda de treino, tempo e parâmetros treináveis. O ranking é determinístico: **maior accuracy, maior F1 macro, menor rank, menor learning rate e ID crescente**. As três primeiras candidatas são identificadas, mas **não avançam automaticamente**.

## Interpretação e eventual CV posterior

O `summary.json` e o CSV registram, para cada candidata, a diferença absoluta de accuracy em relação à baseline atual (`candidate-03`: learning rate `1e-3`, rank `4`, alpha `8`), a distância para a primeira e terceira colocadas e a diferença para a colocada anterior. O resumo destaca ainda as diferenças entre 1ª/2ª, 2ª/3ª e 3ª/4ª posições. Estas são diferenças em pontos de proporção: `0.01` equivale a um ponto percentual.

Um único split não mede a estabilidade da melhoria. A decisão de realizar CV de três folds será tomada após examinar este ranking; não há CV nesta execução. Se aprovado posteriormente, o modo `experiment` existente pode receber os arquivos de `configs/candidate-XX.json` guardados nesta rodada e diretórios de saída novos. O holdout permanece fora da busca; este resultado não promove automaticamente um novo adapter à entrega.

## Artefatos e acompanhamento

Cada execução usa diretórios novos `results/norberto/experiments/grid-search/norberto-lora-grid-<id>/` e `artifacts/norberto-lora-grid-<id>/`. O primeiro contém `summary.json`, `screening_results.csv`, `screening_report.md`, `screening-split.csv`, `configs/`, `screening/`, `run.log` e `pid.txt`. O segundo guarda adapters e checkpoints de screening das nove candidatas. `summary.json` é atualizado após cada candidata e registra commit, hashes, seed, bibliotecas, configuração, estado, ranking e TOP 3 quando a busca terminar. O relatório Markdown e o CSV ordenado, com diferenças relativas à baseline, são produzidos após a nona candidata; resultados parciais ficam preservados em JSON e CSV durante o job.

O modo `--smoke` prepara os diretórios, valida a partição e inicializa um único modelo, sem treinar. A busca completa exige GPU funcional. Para acompanhar uma execução iniciada, use `tail -f results/norberto/experiments/grid-search/norberto-lora-grid-<id>/run.log` e consulte `summary.json`. A interrupção de uma execução em Docker é feita pelo nome do contêiner registrado no comando de lançamento, não pela remoção dos artefatos.

O tempo dos nove treinos depende da GPU e do early stopping; planeje várias horas. O orçamento de 75 minutos continua aplicado a cada treino individual. Se alguma candidata atingir esse limite sem concluir, o job falha explicitamente e **não publica um ranking completo enganoso**; os resultados parciais continuam disponíveis para diagnóstico.

## Resultado da execução de 29/09/2026

O screening [terminou com nove resultados](../../results/norberto/experiments/grid-search/norberto-lora-grid-20260929-170333/screening_results.csv), sobre 12.982 linhas de treino e as mesmas 3.111 linhas de validação em todos os casos. O [resumo reproduzível](../../results/norberto/experiments/grid-search/norberto-lora-grid-20260929-170333/summary.json) e o [relatório tabular](../../results/norberto/experiments/grid-search/norberto-lora-grid-20260929-170333/screening_report.md) guardam hashes, métricas, tempos, épocas e configurações. O TOP 3 é:

| Posição | Candidata | Taxa | Rank / alpha | Acurácia | F1 macro | Ganho sobre a baseline |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `candidate-05` | 0,0003 | 8 / 16 | 44,55% | 44,61% | +6,24 p.p. |
| 2 | `candidate-08` | 0,0003 | 16 / 32 | 44,29% | 44,29% | +5,98 p.p. |
| 3 | `candidate-02` | 0,0003 | 4 / 8 | 43,81% | 43,73% | +5,50 p.p. |

A baseline `candidate-03` (`0,001`, rank 4, alpha 8) ficou em 38,32% de acurácia e 30,09% de F1 macro **neste split**. As diferenças de accuracy foram 0,26 p.p. entre 1ª/2ª, 0,48 p.p. entre 2ª/3ª e 0,93 p.p. entre 3ª/4ª. A vantagem sobre a baseline na mesma validação é relevante como indício, mas não comprova ganho em outros folds. Por isso foi autorizada uma [CV exploratória separada dos três finalistas](cv-finalistas-hiperparametros.md); o screening permanece imutável e nenhuma configuração foi promovida ainda ao modelo de entrega.
