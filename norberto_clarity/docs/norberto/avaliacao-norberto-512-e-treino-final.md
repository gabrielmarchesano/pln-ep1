# Avaliação do NorBERTo-512 e treinamento final

Este relatório contém a **avaliação cruzada no desenvolvimento** e o **treino final sem validação**. A [decisão do modelo escolhido](../modelo-escolhido-norberto-512.md) e a [avaliação separada no holdout](avaliacao-norberto-holdout-final.md) têm documentos próprios.

## Resultado da validação cruzada

A execução `norberto-lora-development-ctx512-20260928` terminou em **98,57 minutos**, com três folds e 16.093 predições OOF. A [auditoria](../../results/norberto/delivery/norberto-lora-development-ctx512-20260928/audit.json) aprovou correspondência dos rótulos, grupos, partições, probabilidades, matrizes de confusão e métricas recalculadas. As 3.999 linhas do holdout congelado ficaram excluídas de treino, tokenização e avaliação desta rodada.

| Modelo | Acurácia OOF | F1 macro | Acurácia balanceada | Acertos |
|---|---:|---:|---:|---:|
| NorBERTo LoRA, contexto 512 | **43,47%** | **43,22%** | 43,46% | 6.995 |
| Baseline com o mesmo treino do encoder | 43,14% | 42,94% | 43,06% | 6.942 |
| Logística lexical com o mesmo treino do encoder | 43,98% | 43,51% | 43,84% | 7.078 |
| Logística lexical no treino externo completo | **43,99%** | **43,54%** | **43,84%** | **7.080** |

O delta de acurácia `NorBERTo − lexical completo` foi **−0,53 ponto percentual**, ou 85 acertos a menos. O intervalo descritivo pareado de 95%, reamostrando os componentes de similaridade, foi **[−1,49; +0,43] p.p.** O intervalo inclui zero; não demonstra superioridade nem equivalência. O campo `resampling_unit` dos JSON herdou o nome `normalized_text_group`, mas os identificadores efetivamente reamostrados nesta rodada são os componentes congelados de similaridade.

Os resultados sustentam a caracterização de desempenho próximo, com vantagem pontual para a logística. **A escolha do NorBERTo para a próxima etapa atende à decisão do projeto de priorizar a contribuição de um encoder pré-treinado adaptado com LoRA e o critério de inovação.** Esta rodada não estabelece que ele seja mais robusto: seu desvio de acurácia entre folds foi 0,90 p.p., contra 0,65 p.p. do lexical completo. Robustez fora da distribuição também não foi medida.

## Folds, seleção de checkpoint e erros

| Fold | Acurácia NorBERTo | F1 macro NorBERTo | Acurácia lexical completo | Melhor época interna | Épocas executadas |
|---|---:|---:|---:|---:|---:|
| 1 | 44,61% | 43,02% | 44,91% | 2 | 3 |
| 2 | 43,45% | 43,15% | 43,67% | 3 | 3 |
| 3 | 42,40% | 42,49% | 43,43% | 2 | 3 |

O NorBERTo ficou abaixo do controle lexical completo nos três folds, com a maior diferença no terceiro. Não houve o colapso de otimização observado na rodada histórica de contexto 256, porém esta execução usa outra partição e outro agrupamento; não cabe atribuir causalmente a diferença ao contexto. A rodada de 256 permanece separada.

| Classe | Recall NorBERTo | Recall lexical completo | F1 NorBERTo |
|---|---:|---:|---:|
| c1 | 46,09% | 41,41% | 45,61% |
| c234 | 33,43% | 33,76% | 35,53% |
| c5 | 50,87% | 56,35% | 48,53% |

O encoder recupera mais exemplos de `c1`, perde recall em `c5` e mantém dificuldade acentuada na classe intermediária. Matriz de confusão, linhas verdadeiras e colunas previstas:

| Classe | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 2.346 | 1.402 | 1.342 |
| c234 | 1.690 | 1.817 | 1.929 |
| c5 | 1.161 | 1.574 | 2.832 |

Os três folds treinaram até a época 3 sem esgotar o orçamento. Os melhores checkpoints foram `checkpoint-1072`, `checkpoint-1647` e `checkpoint-1020`, correspondendo às épocas **2, 3 e 2**. Os hashes dos três arquivos `best_adapter/adapter_model.safetensors` coincidem com os dos respectivos checkpoints escolhidos. Foram 272.643 parâmetros treináveis; o maior pico reservado pelo PyTorch foi 4.456 MiB. O tempo total ficou dentro da estimativa de 90–110 minutos.

Os [resultados completos](../../results/norberto/delivery/norberto-lora-development-ctx512-20260928/results.json), [predições OOF](../../results/norberto/delivery/norberto-lora-development-ctx512-20260928/oof.csv), históricos e checkpoints permanecem preservados. Nenhum resultado histórico foi reescrito para selecionar a próxima etapa.

## Treinamento final concluído

O modo `final` concluiu em **31,18 minutos de treino** (28/09/2026, horário de São Paulo), abaixo da estimativa de 35–45 minutos. Foi treinado um novo modelo a partir de `Itau-Unibanco/NorBERTo-base`, revisão `db73446f89c96044863ea05a39f680524b84bccb`, usando as **16.093 linhas de desenvolvimento** e **duas épocas completas**. A inicialização usou a semente 42 da configuração aprovada. Nenhum adapter dos folds foi carregado. O manifesto registra `mode=final`, `status=complete`, **12.956 grupos de desenvolvimento**, **zero sobreposição** de linhas e grupos com o holdout e `holdout_evaluated=false`.

Não existe época final explicitamente registrada no resultado original. O runner associa o passo de cada `best_checkpoint` ao histórico de validação interna e aplica a **mediana das melhores épocas dos três folds: mediana(2, 3, 2) = 2**. Não usa a acurácia externa para escolher um fold e não confunde a época escolhida com a última época executada. A regra, os valores por fold e os hashes dos arquivos de origem ficam em `epoch_selection` no manifesto final. Para futuros resultados que tragam `final_training_epochs` ou `epoch_selection.final_epochs` explicitamente registrados, esse valor terá prioridade. Na derivação, uma mediana fracionária é arredondada para baixo, com mínimo de uma época.

O treino final não possui validação interna, folds externos, early stopping nem `load_best_model_at_end`. Mantém LoRA, tokenizer, truncamento, taxa de aprendizado, batch, acumulação, regularização, warmup, precisão e atenção da configuração aprovada. O limite de 75 minutos continua como proteção de execução; se for atingido antes das duas épocas, o resultado será marcado como incompleto e a avaliação separada o rejeitará.

A partição é conferida pelos hashes da planilha e do CSV. Verificações adicionais exigem que as linhas e os grupos usados sejam exatamente os do desenvolvimento, com **zero sobreposição** com o holdout. Os identificadores de treino são salvos em `training-rows.csv`, com seu próprio hash. `pilot` e `experiment` mantêm a seleção e os protocolos anteriores; a implementação reutiliza `make_model`, `train_one`, `TokenizedRows`, tokenização, métricas e persistência.

| Modo | Dados usados no otimizador | Seleção de checkpoint | Avaliação |
|---|---|---|---|
| `pilot` | Treino interno do primeiro fold | Validação interna | Validação interna |
| `experiment` | Treino interno de cada fold | Validação interna | Folds externos |
| `final` | Todo o desenvolvimento | Épocas recuperadas do experimento anterior | Nenhuma |
| `evaluate-holdout` | Nenhum; apenas inferência | Nenhuma | Somente o holdout congelado |

Os testes cobriram seleção de épocas, prioridade de registro explícito, configuração divergente, isolamento de dados, treino real de LoRA pequeno sem validação, recarga do adapter e inferência restrita ao holdout. **Suíte completa: 66 aprovados, três pulados.** O diagnóstico numérico da GPU também passou.

## Artefatos e comandos

O adapter final, incluindo a cabeça classificadora e o tokenizer, foi salvo em `artifacts/norberto-lora-final-ctx512-20260928/final_model/` com `trainer.save_model()` e `tokenizer.save_pretrained()`. Esse pacote final e seus metadados foram posteriormente versionados no Git; checkpoints intermediários e adapters experimentais continuam excluídos. Os pesos-base permanecem fora do Git e são baixados na revisão fixada, salvo se já estiverem em `.cache/huggingface/`. O SHA-256 do arquivo `adapter_model.safetensors` é `d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`.

Os [resultados e manifesto](../../results/norberto/delivery/norberto-lora-final-ctx512-20260928/results.json) e `artifacts/norberto-lora-final-ctx512-20260928/run.json` registram modo, protocolo, estado, linhas, isolamento, épocas escolhidas/executadas, perda, tempo, memória GPU, parâmetros treináveis, configuração, revisão e hashes. O treino executou as duas épocas, sem parada por orçamento, com **272.643 parâmetros treináveis**, perda de treino reportada **4,2874**, pico de **3.259 MiB alocados / 3.800 MiB reservados** pelo PyTorch. A perda de treino, medida pelo loop do Trainer, não estima acurácia em dados novos e não é diretamente comparável aos scores de validação dos folds. O holdout **não foi avaliado automaticamente**.

```bash
docker logs --follow --tail 100 pln-norberto-512-final-20260928
```

Ctrl+C fecha apenas o acompanhamento. O comando de treino usa:

```bash
python3 -u -m clarity.finetune final \
  --train data/train.xlsx \
  --config configs/norberto-lora-development-ctx512.json \
  --holdout-splits results/research/dataset-hypotheses-20260923/splits.csv \
  --experiment-dir results/norberto/delivery/norberto-lora-development-ctx512-20260928 \
  --output-dir results/norberto/delivery/norberto-lora-final-ctx512-20260928 \
  --artifact-dir artifacts/norberto-lora-final-ctx512-20260928
```

A avaliação separada foi executada posteriormente com:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --shm-size=2g \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e AMDGPU_TARGETS=gfx1030 \
  -e PYTHONPATH=/workspace/src -e HF_HUB_OFFLINE=1 \
  -v "$PWD:/workspace" -w /workspace pln-rocm:6.4.1 \
  python3 -u -m clarity.finetune evaluate-holdout \
  --train data/train.xlsx \
  --config configs/norberto-lora-development-ctx512.json \
  --holdout-splits results/research/dataset-hypotheses-20260923/splits.csv \
  --artifact-dir artifacts/norberto-lora-final-ctx512-20260928 \
  --output-dir results/norberto/delivery/norberto-lora-final-holdout-20260928
```

Esse comando verificou a origem, o estado completo e os hashes do modelo final, carregou exclusivamente `final_model` e produziu métricas, probabilidades e CSV por linha **sem chamar treinamento**. A [avaliação final do modelo no holdout](avaliacao-norberto-holdout-final.md) obteve 45,31% de acurácia e 45,31% de F1 macro em 3.999 linhas; modo registrado: `evaluate-holdout`. O holdout já havia sido consultado em pesquisas lexicais anteriores, portanto não é historicamente intocado.
