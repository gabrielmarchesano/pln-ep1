# BERTimbau supervisionado — registro experimental

**Registro histórico:** este documento preserva pilotos, hipóteses e decisões na ordem em que ocorreram. As recomendações antigas de contexto 256 não são a escolha vigente. O modelo escolhido após a avaliação completa é o [NorBERTo-512 LoRA](../modelo-escolhido-norberto-512.md); veja também o [índice](README.md).

## Escopo

Em 17/09/2026, após consolidar a MLP, iniciou-se a avaliação do BERTimbau Base supervisionado na NVIDIA MX550 de 2 GB. A AMD RX 6650 XT de 8 GB em outro dispositivo permanece como alternativa futura e não é usada nestas execuções.

Hipótese: adaptar a representação aos rótulos de clareza pode acrescentar informação que um classificador sobre embeddings congelados não aproveitou. Não há garantia de superar os 45,14% da logística lexical. O novo modelo também muda tokenizer e tratamento de textos longos; não é uma ablação exclusivamente do classificador.

Modelo: [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased), revisão `94d69c95f98f7d5b2a8700c420230ae10def0baa`. Pesos públicos baixados para `.cache/huggingface`; nenhum dado do corpus é enviado ao provedor.

## Implementação

- Runner separado: `PYTHONPATH=src .venv/bin/python -m clarity.finetune`. A avaliação clássica existente não foi substituída.
- `AutoModelForSequenceClassification`, três classes, com LoRA em query/value. Rank 4, escala alpha 8, dropout 0,05, cabeça de classificação treinável. Todos os blocos recebem adaptadores na primeira configuração: **149.763 parâmetros treináveis**.
- Caixa preservada na entrada. HTML e espaços são normalizados sem converter para minúsculas. O agrupamento de duplicatas continua usando a normalização original, inclusive minúsculas, para não separar versões equivalentes do mesmo texto.
- Limite inicial de 128 tokens, incluindo CLS/SEP. Para textos maiores, preservar a primeira e a última metade dos tokens (`head_tail`), omitindo o meio; não são janelas independentes. A proporção afetada é medida em `tokenization.json`. Essa aproximação tem custo baixo, mas pode perder informação e criar uma junção artificial entre trechos.
- Cache somente de tokenização, sem ajuste estatístico, com chave incluindo textos sensíveis à caixa, revisão, comprimento, política e versão da normalização. Não reutilizar os embeddings congelados durante fine-tuning.
- AdamW, taxa inicial 0,0002, weight decay 0,01, warmup de 10%, clipping de gradiente em 1. Batch 4 e acumulação 4 (batch efetivo usual 16; o último pode ser menor), FP32 e atenção SDPA.
- Até três épocas; parada após uma época sem melhora de acurácia interna. O checkpoint de maior acurácia interna é restaurado. Guardas interrompem perdas/gradientes não finitos; o orçamento de tempo é verificado após passos do otimizador, não é timeout rígido que inclua toda a avaliação e I/O.

As dependências adicionais estão em `requirements.txt`: PEFT 0.17.1, Accelerate 1.10.1 e psutil 7.2.2. O ambiente usa torch 2.8.0+cu126 e transformers 4.57.1. Os pesos originais permanecem congelados; LoRA e a cabeça são salvos como adaptadores, não como `.joblib` compatível com o comando `predict` clássico.

## Protocolo: não confundir com o nested 3×3 anterior

As divisões externas continuam sendo as três divisões agrupadas com semente 42. Dentro de cada treino externo, a primeira divisão de uma `StratifiedGroupKFold` de cinco folds, semente 42+fold, reserva aproximadamente 20% para seleção de checkpoint. Treino, validação interna e validação externa têm grupos disjuntos.

É **validação externa agrupada com holdout interno de seleção de checkpoint**, não uma busca aninhada 3×3. Não há reajuste em todo o treino externo após escolher épocas: o modelo restaurado usa apenas o subconjunto de otimização. A escolha evita um segundo treinamento caro e é explicitada nos resultados.

Limite da pesquisa adaptativa: escolher uma configuração global a partir do piloto do fold 1 reutiliza rótulos que pertencem aos folds externos 2 e 3. Portanto, uma futura rodada de três folds com essa configuração será **exploratória**, embora cada ajuste individual mantenha seus grupos de treino/validação separados. O fold 1 não participa do piloto BERT, mas já foi avaliado nas famílias anteriores; também não é um teste final inteiramente intocado pela pesquisa. Uma estimativa confirmatória exigiria um teste independente ou repetir toda a seleção de configuração dentro de cada treino externo. Não apresentar intervalos OOF como se incorporassem essa seleção adaptativa.

Dois modos:

1. `pilot`: somente o primeiro treino externo; não prevê nem avalia seu fold externo. A configuração inicial usa um terço agrupado do treino interno. Scores do holdout interno são exploratórios/de seleção, não estimativa de teste.
2. `experiment`: percorre os três folds externos, gera OOF e métricas externas. Deve usar uma configuração própria e registrar se há subsampling ou interrupções por orçamento.

Controles `baseline_matched` e `lexical_matched` são treinados nas mesmas linhas usadas pelo otimizador do encoder. No modo completo, `lexical_full` também usa todo o treino externo com C=0,25, permitindo comparar ao alvo operacional anterior. Nenhum controle de piloto usa rótulos do holdout interno para treinar. O split de cada fold é salvo com números das linhas Excel, sem publicar textos no relatório.

## Teste técnico inicial de GPU

Realizado sem selecionar configurações por acurácia, com rótulos artificiais apenas para exercitar forward/backward. Registros locais: `artifacts/bertimbau-hardware-pilot.json` e `artifacts/bertimbau-hardware-pilot-power.json`.

O primeiro teste encontrou clock de 300 MHz e limite efetivo de 8 W, com sinais de limitação de potência/temperatura reportados pelo driver. Batch 2, 128 tokens, atenção eager e LoRA em todos os blocos:

| Precisão | Segundos por passo após aquecimento | Pico alocado |
|---|---:|---:|
| FP32 | 1,435 | 598,67 MiB |
| FP16 | 3,678 | 727,28 MiB |

Após desativar a economia de energia do sistema, a segunda medição, com atenção SDPA e FP32, foi:

| Batch | Blocos LoRA | Exemplos/s | Pico alocado |
|---|---|---:|---:|
| 4 | Todos | 18,63 | 695,90 MiB |
| 8 | Todos | 17,32 | 930,73 MiB |
| 8 | Apenas os dois últimos | 25,10 | 538,38 MiB |

São benchmarks curtos e não projeções garantidas de execução. Além do estado energético, atenção e batch mudaram, portanto a aceleração não pode ser atribuída exclusivamente a uma dessas alterações. Limites de potência, clocks e proteções térmicas foram mantidos. FP32/batch 4/todos os blocos foi escolhido como ponto inicial com margem de memória; a opção dos dois últimos blocos fica registrada para uma possível ablação posterior.

## Piloto supervisionado inicial

Configuração: [bertimbau-lora-pilot.json](../../configs/research/bertimbau-lora-pilot.json). Até três épocas, orçamento de treinamento de 40 minutos, aproximadamente um terço do treino interno; holdout interno mantido. Estimativa inicial aproximada de 10–20 minutos incluindo validações e controles, condicionada ao perfil energético observado. Não lançar outra grade automaticamente só porque houve capacidade de memória.

```bash
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src .venv/bin/python -u -m clarity.finetune pilot \
  --config configs/research/bertimbau-lora-pilot.json \
  --output-dir results/research/bertimbau-lora-pilot \
  --artifact-dir artifacts/bertimbau-lora-pilot
```

Manifesto, estatísticas de truncamento, splits, métricas por época e predições são registrados. Pesos/checkpoints ficam em `artifacts/` e não entram no Git. Usar diretórios novos para cada configuração; não há retomada automática de experimentos interrompidos. Os adaptadores de cada fold permitem auditoria e carregamento posterior, mas não representam um modelo final treinado em toda a base.

Durante o primeiro piloto, a próxima versão do runner recebeu registro das probabilidades por classe para encoder e controles, tempo de inferência do encoder e validação mais estrita dos valores numéricos da configuração. Essa mudança não altera o treinamento em andamento: o primeiro piloto salva somente rótulos. Rodadas posteriores salvam colunas `prob_<modelo>_<classe>`, com ordem das classes verificada; probabilidades não são consideradas calibradas automaticamente. A auditoria verifica soma, domínio e concordância com os rótulos preditos. Uma combinação de modelos só poderá ter pesos escolhidos na validação interna, nunca no fold externo.

### Diagnóstico de comprimento, antes da segunda rodada

Tokenização sem truncamento, usando a revisão fixada e os textos de entrada (sem rótulos):

| Limite, incluindo CLS/SEP | Respostas truncadas | Fração dos tokens do corpus preservada |
|---|---:|---:|
| 128 | 63,12% | 41,38% |
| 192 | 47,81% | 55,40% |
| 256 | 35,83% | 66,03% |
| 384 | 18,93% | 79,61% |
| 512 | 10,81% | 86,96% |

A mediana é 180 tokens e o percentil 95 é 716. A fração preservada é a razão entre a soma dos comprimentos limitados (sem tokens especiais) e a soma dos comprimentos originais, não a média das frações por resposta. Aumentar contexto tem uma justificativa de cobertura, mas não demonstra ganho preditivo e aumenta o custo. A decisão da próxima rodada depende também da curva de aprendizado do piloto.

## Resultado do primeiro piloto — concluído em 17/09/2026

Arquivos: [results.json](../../results/research/bertimbau-lora-pilot/results.json), [fold-1.json](../../results/research/bertimbau-lora-pilot/fold-1.json), [auditoria](../../results/research/bertimbau-lora-pilot/audit.json). Execução entre 15:18 e 15:39, horário de São Paulo. Código de treinamento iniciado na revisão `39942e7`; os hashes dos fontes no início constam no manifesto. As melhorias posteriores do runner não modificaram esse processo já iniciado.

Treino com 3.514 respostas; holdout interno com 2.617; fold externo de 6.613 respostas **não avaliado**. Os três métodos abaixo usam as mesmas linhas de treinamento e avaliação:

| Modelo | Acurácia interna | F1 macro | Recall c1 | Recall c234 | Recall c5 |
|---|---:|---:|---:|---:|---:|
| BERTimbau + LoRA | 38,33% | 37,33% | 25,52% | 33,79% | 55,56% |
| Baseline palavras + logística | 40,43% | 39,99% | 31,55% | 36,73% | 52,92% |
| Lexical palavras/caracteres + logística C=0,25 | 41,50% | 40,54% | 29,70% | 34,58% | 60,14% |

BERTimbau acertou 1.003 respostas, contra 1.086 do lexical: **−3,17 p.p.**, 83 acertos a menos. O bootstrap pareado de grupos fornece intervalo descritivo de −5,52 a −0,84 p.p., condicionado a estas predições. O holdout também escolheu o checkpoint; isso **não é OOF nem teste independente**. O texto genérico `caveat` do primeiro JSON herdou a palavra “OOF” da função compartilhada; os campos `mode=pilot`, protocolo e arquivos identificam corretamente a avaliação interna. Não comparar diretamente os 38,33% com os 45,14% OOF do lexical completo.

| Época | Perda de validação | Acurácia de validação |
|---|---:|---:|
| 1 | 1,09654 | 35,88% |
| 2 | 1,09321 | 36,72% |
| 3 | 1,08962 | 38,33% |

O melhor checkpoint foi o último, passo 660. Não houve parada antecipada nem interrupção por orçamento. Tempo total registrado: **21,62 min**; treinamento com avaliações por época: **19,45 min**. Pico de memória CUDA alocada: **696,51 MiB**; reservada: **808 MiB** — esses valores não incluem toda a memória de contexto/driver reportada pelo `nvidia-smi`. O treino sustentado reduziu clocks com a temperatura; a estimativa futura usa a execução real, não o benchmark curto.

Os adaptadores foram atualizados e os valores observados foram finitos. O teste real do Trainer terminou, o checkpoint foi salvo/restaurado e a auditoria confirmou linhas, rótulos, grupos, partições e métricas. A suíte atual tem 42 testes aprovados e dois opcionais não executados; `pip check` não encontrou dependências quebradas. O aviso PEFT sobre arquivo de configuração base ausente no modo offline não impediu salvar adaptadores; não confundi-lo com um erro de treinamento.

### Decisão e próxima rodada proposta

Não promover este modelo nem iniciar ainda uma avaliação completa de três folds. A curva melhora até a última época, mas a perda diminui pouco; isso justifica investigar a otimização antes de concluir que a arquitetura atingiu seu limite. O piloto pequeno, o rank baixo e o truncamento também limitam a conclusão. Não há evidência suficiente para apontar uma causa única.

Preparada a configuração [bertimbau-lora-pilot-lr1e3.json](../../configs/research/bertimbau-lora-pilot-lr1e3.json): muda somente a taxa de aprendizado de 0,0002 para 0,001. Mantém partições, amostra, semente, contexto, rank, batch efetivo, épocas e critério de checkpoint. Hipótese: acelerar adaptação em um orçamento curto; a taxa maior também pode desestabilizar ou piorar o resultado. O runner atualizado acrescenta probabilidades e tempo de inferência, sem alterar a função de perda.

Estimativa da segunda rodada: **20–30 minutos**, com orçamento de treino de 40 minutos mais etapas finais. NorBERTo e uma grade completa permanecem fora do escopo desta execução. Se o novo piloto não melhorar, as próximas hipóteses são uma mudança controlada de contexto ou de encoder; se melhorar de forma útil, pode-se avaliar maior volume de treino antes de uma rodada externa cara.

Comando da segunda rodada, usando diretórios separados:

```bash
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src .venv/bin/python -u -m clarity.finetune pilot \
  --config configs/research/bertimbau-lora-pilot-lr1e3.json \
  --output-dir results/research/bertimbau-lora-pilot-lr1e3 \
  --artifact-dir artifacts/bertimbau-lora-pilot-lr1e3
```

Log operacional: `artifacts/logs/bertimbau-lora-pilot-lr1e3.log`. A existência de `results.json` com `status=complete` indica conclusão; só o manifesto/split não basta. Depois de concluir, executar `PYTHONPATH=src .venv/bin/python -m clarity.audit results/research/bertimbau-lora-pilot-lr1e3 --output results/research/bertimbau-lora-pilot-lr1e3/audit.json` e comparar os mesmos 2.617 exemplos com o primeiro piloto. A curva e o tempo precisam ser lidos antes de propor outra rodada.

Segunda rodada executada entre **15:43 e 16:06 de 17/09/2026**, horário de São Paulo, após o commit `4cb7473`.

## Resultado do segundo piloto — taxa 0,001

Arquivos: [results.json](../../results/research/bertimbau-lora-pilot-lr1e3/results.json), [fold-1.json](../../results/research/bertimbau-lora-pilot-lr1e3/fold-1.json) e [auditoria](../../results/research/bertimbau-lora-pilot-lr1e3/audit.json). O split e a tokenização têm os mesmos SHA-256 do primeiro piloto; os controles produziram exatamente os mesmos rótulos. A taxa de aprendizado foi a única alteração experimental entre os JSON de configuração, além da descrição.

| Modelo | Acurácia interna | F1 macro | Acurácia balanceada | Acertos |
|---|---:|---:|---:|---:|
| BERTimbau + LoRA, taxa 0,0002 | 38,33% | 37,33% | 38,29% | 1.003 |
| BERTimbau + LoRA, taxa 0,001 | 39,82% | 39,18% | 39,90% | 1.042 |
| Baseline palavras pareado | 40,43% | 39,99% | 40,40% | 1.058 |
| Lexical palavras/caracteres pareado | 41,50% | 40,54% | 41,47% | 1.086 |

O segundo piloto ganhou **1,49 p.p. e 39 acertos líquidos** sobre o primeiro. Entre as 793 discordâncias de correção, o novo acertou sozinho 416 e o anterior, 377. O intervalo descritivo pareado por grupos foi de **−0,61 a +3,67 p.p.**; portanto, o ganho observado não estabelece superioridade robusta. Contra o lexical pareado, o novo piloto ficou **1,68 p.p. e 44 acertos abaixo**, intervalo de **−3,90 a +0,67 p.p.**.

| Modelo | Recall c1 | Recall c234 | Recall c5 | Predições c1/c234/c5 |
|---|---:|---:|---:|---:|
| BERTimbau, taxa 0,0002 | 25,52% | 33,79% | 55,56% | 531 / 850 / 1.236 |
| BERTimbau, taxa 0,001 | 48,61% | 24,94% | 46,16% | 1.017 / 653 / 947 |
| Lexical pareado | 29,70% | 34,58% | 60,14% | 553 / 825 / 1.239 |

A taxa maior reduziu a preferência por `c5` e deslocou fortemente as predições para `c1`. Isso elevou acurácia e F1 macro, mas piorou os recalls de `c234` e `c5`; não atende ao critério prático anterior de preservar `c234`. A matriz de confusão e o relatório por classe completos estão no JSON.

| Época | Perda de validação | Acurácia | F1 macro |
|---|---:|---:|---:|
| 1 | 1,15268 | 34,05% | 18,66% |
| 2 | **1,08061** | 39,24% | 36,26% |
| 3 | 1,08088 | **39,82%** | **39,18%** |

A primeira época foi instável, seguida de recuperação. O checkpoint da época 3 foi restaurado porque a seleção usa acurácia; a menor perda ocorreu na época 2. O melhor resultado ainda estava na última época, mas o scheduler já havia reduzido a taxa quase a zero, de modo que “continuar o mesmo treino” não é uma extensão bem definida.

Tempo total: **1.346,71 s = 22,45 min**; `Trainer`: **1.213,07 s = 20,22 min**. Não houve parada por orçamento. O pico CUDA foi igual ao anterior: 696,51 MiB alocados e 808 MiB reservados. Gradientes e perdas permaneceram finitos. A auditoria passou e verificou também soma, domínio e argmax das probabilidades.

### Complementaridade exploratória

Uma média simples com pesos iguais entre probabilidades BERTimbau e lexical obteve 41,84%, nove acertos acima do lexical. Ao inspecionar pesos no mesmo holdout, 40% BERTimbau + 60% lexical chegou a 42,15%, 17 acertos acima; intervalo descritivo do delta de −0,89 a +2,32 p.p. Esse peso foi escolhido **depois de observar o holdout usado para checkpoint e configuração**, portanto é apenas diagnóstico de complementaridade e não um resultado selecionado válido. O recall de `c234` do blend de 40% foi 29,82%, abaixo dos 34,58% do lexical. Não promover nem ajustar um ensemble sem seleção interna independente.

### Decisão após os dois pilotos

A otimização era parte do gargalo: uma única mudança de taxa melhorou o resultado, embora com incerteza e forte redistribuição entre classes. Ainda não há evidência para avaliação externa completa nem para atribuir a limitação exclusivamente ao encoder. O piloto usa somente **3.514 das 10.862 linhas disponíveis no treino interno**; a curva também termina no melhor checkpoint de acurácia.

A próxima ablação recomendada é manter taxa 0,001, contexto 128, LoRA e split, alterando apenas `train_subset_folds` de 3 para 1. Isso isola o efeito do volume de treino antes de trocar contexto ou arquitetura. O número de linhas cresce 3,09 vezes; a estimativa conservadora é **50–70 minutos**, incluindo avaliações e controles. Essa execução não foi iniciada. Se não reduzir a diferença para o lexical nem recuperar o equilíbrio por classe, contexto 256 e NorBERTo-base passam a ser alternativas mais justificadas.

## Full inner-training pilot — completed on 22/09/2026

Configuration: [bertimbau-lora-fulltrain-pilot-lr1e3.json](../../configs/research/bertimbau-lora-fulltrain-pilot-lr1e3.json). This run changed only `train_subset_folds` from 3 to 1 relative to the second pilot. It therefore used all 10,862 optimization rows while preserving the same 2,617-row inner holdout, model revision, learning rate, context, LoRA settings, seed, and checkpoint criterion. It ran in `pilot` mode on an AMD Radeon RX 6650 XT through ROCm 6.4 with the documented `gfx1030` override. The 6,613-row outer fold remained untouched.

Files: [results.json](../../results/research/bertimbau-lora-fulltrain-pilot-lr1e3/results.json), [fold-1.json](../../results/research/bertimbau-lora-fulltrain-pilot-lr1e3/fold-1.json), and [audit.json](../../results/research/bertimbau-lora-fulltrain-pilot-lr1e3/audit.json). The audit passed for split isolation, evaluated rows, probabilities, confusion matrices, and recomputed metrics.

| Model | Inner accuracy | Macro F1 | Balanced accuracy | Correct |
|---|---:|---:|---:|---:|
| BERTimbau + LoRA, 3,514 training rows | 39.82% | 39.18% | 39.90% | 1,042 |
| BERTimbau + LoRA, 10,862 training rows | **42.76%** | **41.96%** | **42.82%** | **1,119** |
| Lexical matched, 10,862 training rows | 43.22% | 42.92% | 43.22% | 1,131 |

The larger-training pilot gained **2.94 percentage points and 77 net correct predictions** over the smaller pilot. Of the changed correctness outcomes, the larger pilot alone was correct on 375 rows and the smaller pilot alone on 298. The group-paired descriptive interval for the accuracy delta was **+0.99 to +4.86 percentage points**. This interval is conditional on the repeatedly inspected inner holdout and does not turn the pilot into an independent test.

Against the lexical model trained on the same 10,862 rows, BERTimbau was **0.46 percentage points and 12 correct predictions lower**. The group-paired descriptive interval was **−2.47 to +1.60 percentage points**, so this holdout does not establish a difference between them.

| Model | Recall c1 | Recall c234 | Recall c5 | Predictions c1/c234/c5 |
|---|---:|---:|---:|---:|
| BERTimbau, 3,514 rows | 48.61% | 24.94% | 46.16% | 1,017 / 653 / 947 |
| BERTimbau, 10,862 rows | 43.74% | 27.32% | 57.39% | 849 / 655 / 1,113 |
| Lexical matched, 10,862 rows | 37.70% | 36.28% | 55.67% | 690 / 867 / 1,060 |

The additional data recovered c5 recall and improved c234 recall by 2.38 percentage points relative to the smaller BERTimbau pilot, but c234 remained 8.96 percentage points below the matched lexical control. The model still finished at its best checkpoint, so the result does not demonstrate that the learning curve has saturated.

| Epoch | Validation loss | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 1 | 1.07874 | 40.62% | 36.18% |
| 2 | 1.07623 | 41.23% | 41.24% |
| 3 | **1.07296** | **42.76%** | **41.96%** |

Total runner time was **473.18 seconds (7.89 minutes)**; `Trainer` time including epoch evaluations was **427.18 seconds (7.12 minutes)**. Peak PyTorch memory was 806.36 MiB allocated and 916 MiB reserved. Training completed all three epochs without a budget stop or nonfinite values.

This result supports training-volume sensitivity, but it remains an inner-selection pilot on a holdout already used to choose the learning rate. Do not present 42.76% as OOF or test accuracy. A future three-fold outer run would provide grouped OOF predictions for the fixed configuration, but it would remain exploratory because pilot observations influenced that configuration.

## Context-256 ablation — completed on 22/09/2026

Configuration: [bertimbau-lora-fulltrain-pilot-lr1e3-ctx256.json](../../configs/research/bertimbau-lora-fulltrain-pilot-lr1e3-ctx256.json). Relative to the full inner-training context-128 pilot, this run changed only `max_length` from 128 to 256. It kept the same rows, holdout, seed, learning rate, batch sizes, LoRA adapters, truncation policy, epoch cap, and checkpoint criterion.

The longer context reduced whole-corpus truncation from 63.12% to 35.83% and increased the retained-token fraction from 41.38% to approximately 66.03%. On the 2,617-row inner holdout, the corresponding retained-token fractions were 41.02% and 65.33%.

| Model | Inner accuracy | Macro F1 | Balanced accuracy | Correct |
|---|---:|---:|---:|---:|
| BERTimbau, context 128 | 42.76% | 41.96% | 42.82% | 1,119 |
| BERTimbau, context 256 | **43.41%** | **43.16%** | **43.48%** | **1,136** |
| Lexical matched | 43.22% | 42.92% | 43.22% | 1,131 |

Context 256 gained **0.65 percentage points and 17 net correct predictions** over context 128. It was uniquely correct on 272 rows while context 128 was uniquely correct on 255. The group-paired descriptive interval was **−1.06 to +2.36 percentage points**, so this holdout does not establish a reliable context-length improvement.

Against the matched lexical model, context 256 was **0.19 percentage points and five correct predictions higher**. The two models disagreed in correctness on 801 rows: BERTimbau alone was correct on 403 and lexical alone on 398. The descriptive interval was **−1.85 to +2.35 percentage points** and does not establish superiority.

| Model | Recall c1 | Recall c234 | Recall c5 | Predictions c1/c234/c5 |
|---|---:|---:|---:|---:|
| BERTimbau, context 128 | 43.74% | 27.32% | 57.39% | 849 / 655 / 1,113 |
| BERTimbau, context 256 | 51.97% | 32.65% | 45.82% | 1,033 / 758 / 826 |
| Lexical matched | 37.70% | 36.28% | 55.67% | 690 / 867 / 1,060 |

The longer context improved c234 recall by 5.33 percentage points and c1 recall by 8.24, but reduced c5 recall by 11.57. It produced a more even macro F1, yet c234 remained 3.63 percentage points below the lexical control and c5 moved below it.

The context effect was concentrated in long responses:

| Raw subtoken length | Rows | Context 128 accuracy | Context 256 accuracy | Delta |
|---|---:|---:|---:|---:|
| 0–126 | 956 | 44.67% | 43.93% | −0.73 p.p. |
| 127–254 | 747 | 44.71% | 44.18% | −0.54 p.p. |
| 255+ | 914 | 39.17% | 42.23% | **+3.06 p.p.** |

This pattern is consistent with additional context helping the subgroup that was most severely truncated, but it is a post-hoc subgroup analysis on the same holdout and has no separate uncertainty estimate. Even after the improvement, lexical accuracy for the 255+ group was 43.65%.

| Epoch | Validation loss | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 1 | 1.07229 | 40.77% | 34.23% |
| 2 | **1.07179** | **43.41%** | **43.16%** |
| 3 | 1.07414 | 43.29% | 42.68% |

The epoch-2 checkpoint was restored. Unlike the context-128 run, all validation metrics and loss regressed slightly in epoch 3. The selected context-256 probabilities had log-loss 1.07179 and a 10-bin ECE of 0.0541, compared with 1.05034 and 0.0245 for lexical; probabilities should not be treated as calibrated.

Total runner time was **762.40 seconds (12.71 minutes)** and `Trainer` time was **721.94 seconds (12.03 minutes)**. Peak PyTorch memory was 1,263.19 MiB allocated and 1,520 MiB reserved. The run completed without a budget stop or nonfinite values, and [audit.json](../../results/research/bertimbau-lora-fulltrain-pilot-lr1e3-ctx256/audit.json) passed.

Among the BERTimbau pilots, context 256 is the strongest inner-holdout candidate. Further context or optimization choices on this same holdout would compound adaptive bias. The next defensible model-assessment step is to freeze a configuration before outer evaluation; a three-fold outer run would still be exploratory because this holdout guided the configuration.

## Critérios para iteração

- Primeiro: treinamento finito, gradientes efetivos, margem de VRAM e tempo aceitável. Um `cuda.is_available()` positivo sozinho não basta.
- Avaliar a curva interna e os controles pareados; nunca usar o fold externo para escolher checkpoint. Uma execução de piloto pode indicar ajuste de taxa, regularização, comprimento ou quantidade de dados, mas mudar um fator principal por vez e registrar a justificativa antes da próxima rodada.
- Não extrapolar os scores do piloto como OOF. Rodadas completas são exploratórias quando orientadas pelos resultados anteriores; reportar inclusive tentativas sem melhora.
- Não prometer ganho com mais épocas, maior contexto ou menor regularização. Se o piloto falhar por custo/instabilidade, preservar o registro e testar uma alternativa limitada antes de expandir.
- A decisão final compara acurácia, recall c234, incerteza pareada, custo e estabilidade com o lexical. Sem evidência de ganho, a referência lexical continua disponível.

## Alternativa condicionada aos resultados: NorBERTo-base

Caso a limitação persista após os pilotos controlados, o encoder alternativo considerado é [Itau-Unibanco/NorBERTo-base](https://huggingface.co/Itau-Unibanco/NorBERTo-base). Candidato registrado em 17/09/2026. A revisão pública consultada foi `db73446f89c96044863ea05a39f680524b84bccb`.

Segundo a ficha e a [configuração oficial](https://huggingface.co/Itau-Unibanco/NorBERTo-base/blob/db73446f89c96044863ea05a39f680524b84bccb/config.json), é um ModernBERT de 22 camadas, aproximadamente 150 milhões de parâmetros, dimensão 768, vocabulário 50.368 e máximo arquitetural de 8.192 posições. A ficha declara licença **CC-BY-NC-SA-4.0**. Registrar atribuição e condições da licença em eventual distribuição de adaptadores; não assumir permissão para uso comercial.

O contexto arquitetural máximo não é uma promessa de viabilidade em 2 GB. Antes de treinamento: fixar revisão, verificar tokenizer, executar forward/backward curto, medir memória e tempo, e então começar com 128 ou 256 tokens. O ambiente já contém a implementação ModernBERT no Transformers 4.57.1, mas isso sozinho não comprova compatibilidade completa com PEFT/hardware.

Não basta trocar `model_name` no JSON atual: o ModernBERT usa projeção conjunta `Wqkv`, em vez dos módulos separados query/value do BERT. Também é necessário validar campos do tokenizer, tokens especiais, cabeça de classificação treinável e atenção SDPA, evitando pressupor que todos os encoders aceitam `token_type_ids`. O primeiro comparativo deverá manter partições, tamanho de treino e controles; trocar o encoder também troca tokenizer/arquitetura, portanto não será uma ablação pura de pesos pré-treinados.

Critério: não diagnosticar “gargalo do modelo” apenas por um piloto fraco. Primeiro observar otimização e testar uma alteração principal motivada pela curva; considerar a cobertura de contexto. Se os ajustes limitados não forem promissores, comparar NorBERTo em piloto antes de consumir horas na avaliação externa. Benchmarks publicados em outras tarefas não comprovam superioridade na classificação de clareza do e-SIC.

### NorBERTo pilot contract and ROCm validation

The pilot was registered on 2026-09-22 before the long run. Configuration: [norberto-lora-fulltrain-pilot-lr1e3-ctx256.json](../../configs/research/norberto-lora-fulltrain-pilot-lr1e3-ctx256.json). It preserves fold 1, the 10,862/2,617 inner split, 256-token head-tail policy, rank 4, alpha 8, dropout 0.05, batch 4, four accumulation steps, learning rate 0.001, three epochs, patience 1, seed 42, and matched controls. The deliberate experimental change is the pinned encoder and its tokenizer. Because tokenization also changes, this is a controlled pipeline comparison rather than a weight-only ablation.

The installed tokenizer declares only `input_ids` and `attention_mask`. The runner now creates `token_type_ids` conditionally. LoRA targets the joint `Wqkv` projection in all 22 layers; the sequence-classification head remains trainable. ModernBERT reference compilation is explicitly disabled because the ROCm image has no C compiler for Triton; SDPA remains enabled. This avoids an environmental compilation failure without changing model weights or attention semantics.

A three-step batch-4, context-256 forward/backward/optimizer smoke test completed on the RX 6650 XT. The loaded model had 149,879,814 parameters and 272,643 trainable parameters. Losses were finite. Peak PyTorch memory was 1,658.73 MiB allocated and 1,740 MiB reserved. After the first warm-up step, steps took 0.123 and 0.127 seconds. Based on this measurement and the 12.71-minute BERTimbau context-256 run, the complete pilot is estimated at 20–25 minutes, with a hard training budget of 60 minutes.

The pilot is considered worth expanding only if its paired results improve the context-256 BERTimbau point estimate by a practically relevant margin (provisionally 1 percentage point in accuracy) without reducing macro F1 or `c234` recall, while remaining stable and operationally affordable. A paired interval excluding zero would be stronger evidence, but the single internal holdout is primarily for model selection and cannot establish an external generalization gain.

### NorBERTo pilot results

The run completed on 2026-09-22. Files: [results.json](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx256/results.json), [fold-1.json](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx256/fold-1.json), [predictions](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx256/inner-predictions.csv), and [audit.json](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx256/audit.json). The audit passed row identity, grouped split isolation, probabilities, confusion matrices, and recomputed metrics.

| Model | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| NorBERTo LoRA | **44.71%** | **44.36%** | **44.73%** |
| BERTimbau LoRA, context 256 | 43.41% | 43.16% | 43.48% |
| Matched lexical | 43.22% | 42.92% | 43.22% |
| Matched baseline | 43.22% | 43.17% | 43.23% |

NorBERTo gained **1.30 percentage points and 34 net correct predictions** over BERTimbau. NorBERTo alone was correct on 310 rows and BERTimbau alone on 276; their labels differed on 810 of 2,617 rows. The normalized-text-group bootstrap interval for the accuracy delta was **−0.61 to +3.23 percentage points**. Against the matched lexical model, NorBERTo gained **1.49 points and 39 net correct predictions**, with an interval of **−0.48 to +3.58 points**. Both intervals include zero and are descriptive on a repeatedly inspected inner holdout.

| Model | c1 recall | c234 recall | c5 recall | Correct c1 / c234 / c5 |
|---|---:|---:|---:|---:|
| NorBERTo | 42.92% | 34.81% | **56.47%** | 370 / 307 / 493 |
| BERTimbau context 256 | **51.97%** | 32.65% | 45.82% | 448 / 288 / 400 |
| Matched lexical | 37.70% | **36.28%** | 55.67% | 325 / 320 / 486 |

Relative to BERTimbau, NorBERTo increased `c234` recall by 2.15 points and `c5` recall by 10.65, but reduced `c1` recall by 9.05. Relative to lexical, it improved `c1` and `c5`, while `c234` remained 1.47 points lower. The aggregate gain is therefore a substantial redistribution among classes, not uniform dominance.

| Epoch | Validation loss | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 1 | 1.08929 | 40.50% | 36.93% |
| 2 | **1.06783** | 42.76% | 42.45% |
| 3 | 1.07366 | **44.71%** | **44.36%** |

The epoch-3 checkpoint was selected by the registered accuracy criterion. Accuracy and macro F1 were still improving, but validation loss worsened after epoch 2. This divergence is compatible with less reliable probabilities even when argmax classification improves. NorBERTo had log-loss 1.07366, 10-bin ECE 0.0424, and multiclass Brier score 0.6470. BERTimbau had 1.07179, 0.0541, and 0.6479; lexical had the best probability scores at 1.05034, 0.0245, and 0.6354. None of these probabilities should be assumed calibrated.

The gain was present across BERTimbau-token length strata, but was largest in the middle range:

| Raw BERTimbau subtokens | Rows | NorBERTo | BERTimbau | Lexical | NorBERTo − BERTimbau |
|---|---:|---:|---:|---:|---:|
| 0–126 | 956 | 44.25% | 43.93% | 42.99% | +0.31 p.p. |
| 127–254 | 748 | 46.93% | 44.12% | 43.05% | **+2.81 p.p.** |
| 255+ | 913 | 43.37% | 42.28% | 43.59% | +1.10 p.p. |

The tokenizers had nearly identical aggregate coverage at context 256: NorBERTo truncated 7,181 of all 20,092 rows and BERTimbau 7,198. On the evaluated rows, they truncated 914 and 913 respectively. Consequently, the gain cannot be attributed to a materially larger retained-token budget. These length comparisons are post-hoc and have no separate uncertainty interval.

Total runner time was **1,205.97 seconds (20.10 minutes)** and `Trainer` time was **1,147.50 seconds (19.12 minutes)**. Inference took 39.13 seconds. Peak PyTorch memory was 1,663.31 MiB allocated and 1,932 MiB reserved. Relative to BERTimbau context 256, total time increased by 58.2%, training time by 58.9%, and reserved memory by 27.1%. The run completed all three epochs without budget exhaustion or nonfinite values.

The pilot meets the preregistered point-estimate threshold: accuracy improved by more than one point, macro F1 and `c234` recall also improved, and the operational cost remained acceptable. It is therefore promising enough to freeze this configuration for a broader evaluation. It is not evidence of a confirmed gain: the paired intervals include zero, fold 1 was used for checkpoint selection and repeated architecture decisions, and `c1` recall degraded materially. A three-fold outer run would provide broader coverage but must remain labeled exploratory because this configuration was selected after inspecting fold 1.

### NorBERTo context-512 ablation contract

The context-512 pilot was registered before execution on 2026-09-22. Configuration: [norberto-lora-fulltrain-pilot-lr1e3-ctx512.json](../../configs/research/norberto-lora-fulltrain-pilot-lr1e3-ctx512.json). It changes only `max_length` from 256 to 512 relative to the completed NorBERTo pilot. The model revision, inner split, training size, head-tail policy, LoRA configuration, optimizer, learning-rate schedule, batches, epochs, checkpoint metric, and seed remain fixed. The 75-minute budget is an operational safety limit and is not intended to stop a normal run.

At 512 tokens, full-corpus truncation is expected to fall from 7,181/20,092 rows (35.74%) to 2,192/20,092 (10.91%). On the inner holdout, 609 responses currently truncated at 256 tokens fit entirely within the larger budget; 305 remain truncated. A batch-4 training and batch-8 evaluation ROCm smoke test completed with finite losses. Warm training steps took approximately 0.313 seconds, with 3,380 MiB peak reserved memory during training and 3,572 MiB observed after evaluation. Estimated total runtime is 48–55 minutes.

The provisional practical criterion is at least +0.5 percentage points in accuracy over NorBERTo context 256, without reducing macro F1 or `c234` recall. The criterion is descriptive rather than confirmatory: this is another adaptively selected comparison on fold 1. A result below the criterion is retained and reported rather than followed by unbounded context tuning.

### NorBERTo context-512 results

The run crossed midnight and completed on 2026-09-23 local time. Files: [results.json](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx512/results.json), [fold-1.json](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx512/fold-1.json), [predictions](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx512/inner-predictions.csv), and [audit.json](../../results/norberto/experiments/norberto-lora-fulltrain-pilot-lr1e3-ctx512/audit.json). The audit passed row identity, grouped partitions, probabilities, confusion matrices, and recomputed metrics.

| Model | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| NorBERTo, context 256 | **44.71%** | **44.36%** | **44.73%** |
| NorBERTo, context 512 | 44.52% | 44.06% | 44.55% |
| BERTimbau, context 256 | 43.41% | 43.16% | 43.48% |
| Matched lexical | 43.22% | 42.92% | 43.22% |

Context 512 lost **0.19 percentage points and five net correct predictions** relative to context 256. Context 512 alone was correct on 213 rows and context 256 alone on 218; their predicted labels differed on 587 of 2,617 rows. The normalized-text-group bootstrap interval for the context effect was **−1.80 to +1.37 percentage points**. It includes zero and does not support a reliable improvement.

Context 512 remained above BERTimbau by 1.11 points and above lexical by 1.30 points, but both descriptive intervals included zero. The relevant model-selection comparison is against NorBERTo context 256, which was the incumbent configuration before this ablation.

| Model | c1 recall | c234 recall | c5 recall | Correct c1 / c234 / c5 |
|---|---:|---:|---:|---:|
| NorBERTo context 512 | 43.50% | 32.65% | **57.50%** | 375 / 288 / 502 |
| NorBERTo context 256 | 42.92% | **34.81%** | 56.47% | 370 / 307 / 493 |
| Matched lexical | 37.70% | 36.28% | 55.67% | 325 / 320 / 486 |

The larger context gained 0.58 points of `c1` recall and 1.03 points of `c5`, but lost **2.15 points of `c234` recall**. It therefore failed both the accuracy and class-balance parts of the registered criterion.

| Raw NorBERTo subtokens | Rows | Context 512 | Context 256 | Lexical | Context effect |
|---|---:|---:|---:|---:|---:|
| 0–254 | 1,703 | 44.74% | 45.04% | 42.98% | −0.29 p.p. |
| 255–510 | 609 | 42.69% | **44.33%** | 42.69% | −1.64 p.p. |
| 511+ | 305 | **46.89%** | 43.61% | 45.57% | +3.28 p.p. |

The additional context helped the longest 305 responses, producing ten net additional correct predictions in that subgroup. This was offset by ten fewer correct predictions among the 609 responses that newly fit without truncation and five fewer among shorter responses. More retained text is therefore not uniformly useful; the middle-length decline is evidence against treating reduced truncation as an automatic quality improvement. These are post-hoc subgroup results without independent uncertainty estimates.

| Epoch | Validation loss | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 1 | 1.07073 | 42.76% | 39.39% |
| 2 | **1.06672** | 43.87% | 41.97% |
| 3 | 1.06737 | **44.52%** | **44.06%** |

The epoch-3 checkpoint was selected by accuracy. Context 512 learned faster initially than context 256, but did not finish with a better classification checkpoint. Its final log-loss improved from 1.07366 to 1.06737 and multiclass Brier score from 0.6470 to 0.6439, while 10-bin ECE worsened from 0.0424 to 0.0520. The lexical model retained the best probability quality. The mixed calibration result does not offset the lower accuracy, macro F1, and `c234` recall.

Total runner time was **2,326.98 seconds (38.78 minutes)**, compared with 20.10 minutes for context 256: a **93.0% increase**. Training took 36.84 minutes, inference 98.01 seconds, and peak PyTorch memory was 3,260.54 MiB allocated and 4,040 MiB reserved. Reserved memory increased by 109.1% relative to context 256. The run completed without budget exhaustion or nonfinite values.

Conclusion: context 512 is technically viable but does not justify its cost for the general classifier. It failed the registered +0.5-point criterion, reduced macro F1 and `c234` recall, nearly doubled runtime, and more than doubled reserved memory. **NorBERTo context 256 remains the recommended frozen configuration.** The positive 511+ subgroup result may motivate a future length-routed system, but selecting and validating such a policy would require a separate protocol and must not be inferred from this inspected holdout alone.

### Frozen NorBERTo three-fold evaluation contract

The exploratory outer evaluation was authorized and registered before execution on 2026-09-23. Configuration: [norberto-lora-fulltrain-experiment-lr1e3-ctx256.json](../../configs/research/norberto-lora-fulltrain-experiment-lr1e3-ctx256.json). It freezes the selected context-256 model revision, tokenization, LoRA configuration, optimization, seed, and three-epoch checkpoint protocol. No fold result may change the remaining configuration.

Each of the three grouped outer folds is evaluated once. Within each outer-training partition, the first deterministic grouped inner split supplies the training rows and checkpoint-selection holdout. The encoder is not refit on the complete outer-training partition after checkpoint selection. Matched controls use the encoder's smaller optimizer-visible subset; `lexical_full` uses the complete outer-training partition. This is grouped outer cross-validation with one inner holdout, not nested inner k-fold hyperparameter search.

The estimated total runtime is 65–75 minutes on the RX 6650 XT via Docker/ROCm. The 60-minute budget applies independently to each fold and is a safety limit; expected training time is approximately 19 minutes per fold. Results remain exploratory because fold 1 guided the configuration before this outer run. The evaluation improves coverage across all rows but cannot retroactively make fold 1 an untouched confirmatory test.

### Frozen NorBERTo three-fold evaluation results

The run completed on 2026-09-23. Files: [results.json](../../results/norberto/experiments/norberto-lora-fulltrain-experiment-lr1e3-ctx256/results.json), [OOF predictions](../../results/norberto/experiments/norberto-lora-fulltrain-experiment-lr1e3-ctx256/oof.csv), per-fold reports, and [audit.json](../../results/norberto/experiments/norberto-lora-fulltrain-experiment-lr1e3-ctx256/audit.json). The audit passed all 20,092 external rows, grouped partition isolation, probability columns, confusion matrices, and recomputed metrics.

| Model | Pooled accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| NorBERTo LoRA | 41.52% | 41.45% | 41.42% |
| Matched lexical | 44.70% | 44.50% | 44.64% |
| Full outer-train lexical | **45.14%** | **44.91%** | **45.09%** |
| Matched baseline | 43.62% | 43.56% | 43.61% |

NorBERTo lost **3.18 percentage points and 638 net correct predictions** against the matched lexical control, with a grouped descriptive interval of **−3.99 to −2.38 points**. Against the full outer-train lexical reference, it lost **3.61 points and 726 net correct predictions**; the interval was **−4.45 to −2.76 points**. Both intervals are entirely negative.

| Fold | NorBERTo accuracy | NorBERTo F1 | Full lexical accuracy | Best inner accuracy | Epochs |
|---|---:|---:|---:|---:|---:|
| 1 | 44.43% | 43.63% | 45.91% | 44.71% | 3 |
| 2 | **36.39%** | **35.44%** | 44.86% | 35.47% | 2 |
| 3 | 43.90% | 43.54% | 44.65% | 44.38% | 3 |

Fold 2 exposed a serious optimization failure. Late in epoch 1, finite pre-clipping gradient norms rose from ordinary single/double-digit values to hundreds and above 1,000. Inner accuracy reached only 35.47%, epoch 2 fell to 34.12% with 23.58% macro F1, and patience-1 early stopping restored epoch 1. The external model over-predicted `c234` and achieved 36.39%. The failure is associated with this split/seed at learning rate 0.001; this run alone cannot separate split composition from initialization/order effects.

The sensitivity analysis excluding fold 2 is descriptive and does not remove it from the official result. Even on folds 1 and 3, NorBERTo reached 44.16% versus 45.28% for full lexical: **−1.12 points**, with interval **−2.03 to −0.26 points**. Thus the aggregate deficit is amplified by fold 2 but not created solely by it. NorBERTo fold accuracy had a 3.67-point standard deviation, versus 0.55 for full lexical.

| Model | c1 recall | c234 recall | c5 recall |
|---|---:|---:|---:|
| NorBERTo | 37.70% | **39.11%** | 47.45% |
| Full lexical | **44.02%** | 36.45% | **54.80%** |

NorBERTo improved pooled `c234` recall by 2.66 points but lost 6.32 points on `c1` and 7.36 on `c5`. Much of the `c234` gain came from the unstable fold-2 prediction shift, so it does not establish uniformly better intermediate-class recognition.

Probability quality was also worse: NorBERTo log-loss 1.1061, 10-bin ECE 0.0772, and multiclass Brier 0.6671, versus 1.0381, 0.0117, and 0.6272 for full lexical. Fold 2 alone had log-loss 1.2081 and ECE 0.1557.

Total runtime was **3,518.22 seconds (58.64 minutes)**: 50.30 minutes of training and 5.01 minutes of encoder inference. Peak reserved memory ranged from 1,926 to 1,932 MiB. No fold hit its time budget or produced NaN/Inf; finite but extreme gradients and early stopping are part of the observed instability, not an infrastructure failure.

This remains an exploratory outer evaluation because fold 1 informed configuration selection and rows later used as folds 2/3 outer tests influenced that selection. Nevertheless, it covers every row exactly once externally and provides strong negative evidence about this frozen pipeline.

Conclusion: **do not promote NorBERTo LoRA at learning rate 0.001 as the final classifier.** The lexical model remains the operational reference. A lower-rate NorBERTo experiment could test whether optimization can be stabilized, but it would be a new adaptively motivated experiment and should not override this negative result.
