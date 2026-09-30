# Registro de desenvolvimento — EP1

Este é o **registro cronológico do desenvolvimento**. Para localizar a decisão vigente sem percorrer o histórico, comece pela [página do NorBERTo-512 escolhido](modelo-escolhido-norberto-512.md); o [índice](README.md) separa avaliações de experimentos. As seções abaixo preservam métodos, resultados e decisões **à época**; recomendações antigas não substituem a escolha final.

**Decisão de dados mantida desde 23/09/2026:** seguir com `data/train.xlsx` e seus rótulos originais. As rodadas de tratamento foram concluídas e auditadas sem ganho estável; seus executores ficam em `src/model_research/dataset/`, separados do pacote de produção. O protocolo posterior reservou 16.093 linhas de desenvolvimento e 3.999 de holdout fixo; a CV e o treino final do NorBERTo usaram apenas o desenvolvimento. Como o holdout foi consultado repetidamente, constitui validação fixa, não teste final independente.

## Decisão vigente — 29/09/2026

### Busca posterior de hiperparâmetros — screening e CV dos finalistas concluídos

Sem alterar o NorBERTo-512 já entregue, foram avaliadas [nove configurações LoRA](norberto/screening-hiperparametros.md) em um único split fixo e agrupado das 16.093 linhas de desenvolvimento. A baseline atual (`learning_rate=0,001`, rank 4, alpha 8) marcou 38,32% de acurácia nesse split. Os três melhores foram `candidate-05` (rank 8, taxa 0,0003; 44,55%), `candidate-08` (rank 16, taxa 0,0003; 44,29%) e `candidate-02` (rank 4, taxa 0,0003; 43,81%). A diferença entre 1º e 2º foi de apenas 0,26 p.p.; a validação única não permite concluir estabilidade.

Por autorização posterior, foi executada uma [CV separada dos três finalistas](norberto/cv-finalistas-hiperparametros.md), usando o modo `experiment` original com três folds para cada configuração. `candidate-08` venceu com **43,73% de acurácia média e 42,81% de F1 macro médio**; `candidate-02` obteve 43,17% / 42,32%, e `candidate-05`, 41,23% / 39,22%. O vencedor supera a configuração anterior nos mesmos folds em apenas 0,25 p.p. de acurácia OOF agregada, com intervalo descritivo pareado [−0,37; +0,84] p.p.; não há melhoria estável demonstrada. Seu F1 macro OOF caiu 0,13 p.p. e o recall de `c234` também caiu. As três auditorias de OOF e splits passaram, sem uso do holdout. Pela métrica primária predefinida, `candidate-08` foi selecionado para um novo treino final de **três épocas**, mediana das melhores épocas de fold (2, 3, 3). Os relatórios de screening e CV permanecem preservados; o adapter e a planilha de entrega existentes não são substituídos automaticamente.

O [treino final exploratório de `candidate-08`](norberto/treino-final-candidate-08.md) terminou em 30/09/2026: as três épocas foram executadas nas 16.093 linhas de desenvolvimento em **47,23 minutos**, sem parada por orçamento ou validação. O adapter PEFT/LoRA foi salvo em `artifacts/norberto-lora-grid-final-20260930-130937/final_model/`; hashes e recarga com inferência sintética foram conferidos. A perda de treino foi 4,5884, mas **não é uma estimativa de acurácia**. O próprio treino não avaliou o holdout; a avaliação separada ocorreu depois.

Posteriormente, o comando separado `evaluate-holdout` [avaliou esse adapter salvo](norberto/avaliacao-holdout-candidate-08.md) nas 3.999 linhas congeladas, sem novo treino: **35,98% de acurácia e 35,31% de F1 macro**, contra **45,31% / 45,31%** do adapter anterior nos mesmos IDs de linha. Foram **373 acertos a menos**; o intervalo descritivo pareado para a diferença de acurácia foi [−11,80; −6,94] p.p. A auditoria de linhas, probabilidades, métricas e hashes passou, e a recarga de um adapter de fold reproduziu exatamente suas predições de CV. A causa da deterioração no treino final não foi isolada. **Decisão: não promover `candidate-08`; manter o adapter e `deliveries/test1.xlsx` anteriores.** Essa avaliação reutiliza um holdout já consultado, portanto não deve orientar novas escolhas de época ou hiperparâmetros como se fosse teste independente.

### NorBERTo-512 escolhido — avaliação, treino e holdout concluídos

A [avaliação auditada dos três folds](norberto/avaliacao-norberto-512-e-treino-final.md) terminou em **98,57 minutos**: NorBERTo obteve **43,47% de acurácia e 43,22% de F1 macro**, contra **43,99% e 43,54%** da logística lexical completa nos mesmos folds. A diferença de −0,53 p.p. tem intervalo descritivo [−1,49; +0,43] p.p. O projeto escolheu o NorBERTo pela contribuição de arquitetura pré-treinada com LoRA e pelo objetivo de inovação; a rodada não demonstra maior robustez ou superioridade preditiva. Os três adaptadores foram salvos e conferidos contra os checkpoints selecionados.

O modo `final` treinou **um novo NorBERTo base nas 16.093 linhas de desenvolvimento**, por **duas épocas**, escolhidas automaticamente pela mediana das melhores épocas internas dos folds (2, 3, 2). Não houve folds, conjunto de validação nem early stopping. A configuração restante foi preservada. O adapter foi salvo em `artifacts/norberto-lora-final-ctx512-20260928/final_model/`; treino concluído em **31,18 minutos** e recarga auditada. Posteriormente, o modo separado [evaluate-holdout](norberto/avaliacao-norberto-holdout-final.md) apenas carregou o modelo salvo e avaliou as 3.999 linhas congeladas, sem treinamento: **45,31% de acurácia e 45,31% de F1 macro**, contra **45,09% e 44,78%** da logística treinada apenas no desenvolvimento. A diferença de nove acertos não demonstra superioridade geral; o recall de `c234` foi 42,70% contra 35,00% do lexical. A [decisão documentada](modelo-escolhido-norberto-512.md) distingue esse modelo da referência lexical. Suíte completa no encerramento do treino: **66 testes aprovados, três pulados**.

A entrega de [`data/test1.xlsx` com NorBERTo-512](norberto/entrega-test1-norberto-512.md) foi concluída posteriormente: **900 etiquetas `clarity`** em `deliveries/test1.xlsx` (versionada), com entrada intacta, demais células e estilos preservados, probabilidades verificadas e hashes registrados. O modo `predict-test` não treinou nem usou o teste para seleção. A execução pública é **direta em Python**, com seleção `--device auto/cpu/gpu`; o Docker/ROCm e os overrides de GPU pertencem somente ao ambiente local. Uma inferência real em CPU de duas linhas reproduziu as classes da execução AMD nessas linhas; NVIDIA usa a API PyTorch comum, sem validação com este adapter nesta máquina. A suíte completa após a portabilidade teve **73 testes aprovados, três pulados**. Como não há gabarito em `test1.xlsx`, não se calculou acurácia ou F1 para essa planilha.

A reorganização manteve `src/clarity/` para o NorBERTo e utilitários comuns; modelos clássicos, semânticos e tratamentos de dados foram deslocados para `src/model_research/`, com configurações em `configs/research/`, testes em `tests/research/` e relatórios em `docs/research/`. Posteriormente, os resultados físicos foram separados em [`results/research/`](../results/research/README.md) e [`results/norberto/`](../results/norberto/README.md), com a entrega vigente em `results/norberto/delivery/` e a busca de hiperparâmetros não promovida em `results/norberto/experiments/grid-search/`. Os links simbólicos temporários foram removidos; leitores de resultados resolvem referências históricas gravadas nos manifests sem reescrever resultados ou hashes. O README anterior foi arquivado. O ponto de entrada principal oferece `train` (novo adapter, somente desenvolvimento) e `test` (preencher Excel sem treinar); a CLI histórica continua acessível como módulo de pesquisa. A suíte passou com 73 testes e três pulados após a separação inicial.

Na execução atual, os modos de treino, inferência e avaliação mostram no terminal o dispositivo selecionado (`cpu` ou `cuda`), o backend real (`CPU`, `CUDA` ou `ROCM`) e o nome da GPU, quando disponível. Recomendamos fortemente GPU NVIDIA/CUDA ou AMD/ROCm para acelerar o NorBERTo, sobretudo no treinamento; a CPU permanece como alternativa mais lenta. Essa observabilidade não altera os splits, os hiperparâmetros nem os resultados históricos.

### Resultados anteriores

O desenvolvimento passou da comparação de famílias para uma investigação da qualidade dos dados. A base original e seus rótulos permanecem intactos; o pedido do cidadão e os anexos citados nas respostas não estão disponíveis e não serão usados. A primeira [avaliação do SVM no holdout fixo](research/avaliacao-svm-holdout-fixo.md) obteve **44,81% de acurácia e 44,47% de F1 macro** em 3.999 linhas, após seleção em três folds das 16.093 linhas de desenvolvimento. Esse resultado inaugura uma validação fixa reutilizada, não um teste final independente. Os relatórios vinculados abaixo preservam protocolos, métricas e ressalvas.

### Rodada prospectiva de hipóteses do dataset — concluída

A [rodada de oito políticas](research/experimento-hipoteses-dataset.md) terminou em 16,07 minutos e foi [auditada](../results/research/dataset-hypotheses-20260923/audit.json). O holdout de 3.999 linhas foi congelado por componentes de similaridade textual e **não avaliado**; 16.093 linhas de desenvolvimento foram usadas nos mesmos três folds para todas as ablações. A planilha original e os modelos de produção permanecem intactos. O holdout é prospectivo, não historicamente independente, pois toda a base já havia sido analisada nas rodadas anteriores.

O controle LinearSVC atingiu **44,35% de acurácia e 43,73% de F1 macro** nestes novos folds. A melhor pontuação pontual veio da combinação de normalização ampliada, atributos de estilo, voto majoritário em duplicatas exatas e filtro por consenso OOF: **44,50% / 43,85%**, apenas 25 acertos líquidos a mais. O delta de acurácia foi +0,16 p.p., com intervalo descritivo pareado de 95% **[−0,34; +0,62] p.p.**; dois folds ganharam, o terceiro perdeu 0,65 p.p., e o recall de `c234` caiu de 32,21% para 31,07%. Estilo isolado ganhou sete acertos; maioria, normalização, filtro por consenso e classificação ordinal não melhoraram o controle. A variante ordinal perdeu 120 acertos.

**Decisão após essa rodada:** usar a planilha e os rótulos originais, sem mascaramento, correção automática de duplicatas, filtro por consenso ou atributos experimentais; não alterar o pipeline de produção. Os executores dessas hipóteses ficam isolados em [src/model_research/dataset](../src/model_research/dataset/README.md). Na época, a avaliação do holdout ficou reservada; ela ocorreu posteriormente no SVM lexical. O score absoluto desta divisão não deve ser comparado diretamente aos 45,26% da rodada de limpeza nem aos 45,18% históricos.

### Resultados que motivaram a mudança de foco

| Etapa | Protocolo e resultado | Decisão |
|---|---|---|
| Modelos lexicais | LinearSVC palavras/caracteres, C=0,05: **45,18% OOF agregados** (45,19% de média dos folds). Logística lexical, C=0,25: **45,14%**. Apenas nove acertos de diferença em 20.092 linhas; intervalo pareado inclui zero. | Usar o SVM como maior score pontual no teste de limpeza, sem declarar superioridade sobre a logística. |
| MiniLM congelado + MLP | MLP: **43,68%**; procedimento `selected` escolheu a logística lexical nos três folds e obteve **45,14%**. | Não promover a MLP. |
| BERTimbau LoRA | Pilotos internos: **38,33%** e **39,82%** com treino reduzido; **42,76%** com treino interno completo e contexto 128; **43,41%** com contexto 256. | O aumento do treino e do contexto ajudou o piloto, mas não gerou avaliação externa confirmatória. |
| NorBERTo LoRA | Piloto interno, contexto 256: **44,71%**; contexto 512: **44,52%**, com quase o dobro do tempo. Avaliação exploratória OOF de 256: **41,52%**, contra **45,14%** do lexical completo e **44,70%** do lexical pareado por volume de treino. | Não promover o encoder nem aumentar contexto automaticamente. |

Scores de pilotos internos medem seleção/checkpoint em um holdout, não são comparáveis diretamente a OOF. No experimento NorBERTo completo, o fold 2 caiu para 36,39% com instabilidade de otimização; excluir esse fold *depois* de observá-lo não inverte a conclusão. O intervalo descritivo da diferença NorBERTo menos lexical completo ficou abaixo de zero. A configuração do encoder foi escolhida após pilotos, então seus três folds continuam exploratórios. Consulte [experimento-bertimbau.md](research/experimento-bertimbau.md) e os [resultados NorBERTo](../results/norberto/experiments/norberto-lora-fulltrain-experiment-lr1e3-ctx256/results.json).

### Diagnóstico do dataset e decisão de limpeza

A auditoria confirmou **316 grupos de respostas idênticas com rótulos conflitantes**, envolvendo **1.975 linhas**. Mesmo um classificador que memorizasse o rótulo majoritário de cada texto da própria amostra teria **919 divergências mínimas**; isso não explica sozinho a acurácia de ~45%. A classe `c234`, que agrega notas 2–4, teve recall lexical de **36,45%**. Respostas repetidas e menções a anexos sugerem informação ausente, mas não permitem inferir qual rótulo deveria prevalecer. O [relatório do dataset](research/analise-problemas-dataset.md) separa fatos medidos de hipóteses.

Foi implementada uma [rodada controlada de limpeza](research/experimento-limpeza-dataset.md) com o LinearSVC de C=0,05 e quatro políticas: texto original, mascaramento de URLs/e-mails/números longos, exclusão de grupos com rótulos conflitantes **somente no treino de cada divisão**, e combinação. Três folds externos e três internos usam agrupamento pelo texto mascarado; a política é escolhida internamente e comparada com o controle original nos mesmos folds. Nenhum rótulo de validação participa da filtragem; a fonte `data/train.xlsx` não é editada. A mudança de agrupamento impede comparar diretamente o score absoluto com os 45,18% históricos.

**Resultado concluído e auditado:** a rodada terminou em 12,30 minutos na CPU do contêiner ROCm. O [resultado completo](../results/research/dataset-cleaning-svc-20260923/results.json), as [20.092 predições OOF](../results/research/dataset-cleaning-svc-20260923/oof.csv) e a [auditoria independente](../results/research/dataset-cleaning-svc-20260923/audit.json) estão registrados. A suíte passou antes do início: 48 testes aprovados, três pulados.

| Política | Acurácia OOF agregada | F1 macro | Acertos | Diferença de acertos contra `raw` |
|---|---:|---:|---:|---:|
| `raw` | **45,26%** | 44,99% | 9.094 | Referência |
| `masked` | 44,95% | 44,69% | 9.031 | −63 |
| `conflicts_excluded` | 45,10% | 44,69% | 9.061 | −33 |
| `masked_conflicts_excluded` | 44,79% | 44,41% | 9.000 | −94 |
| `selected` | 45,16% | 44,89% | 9.073 | −21 |

A seleção interna escolheu `raw` nos folds 1 e 2 e `masked` no fold 3. A diferença primária `selected − raw` foi **−0,10 ponto percentual** (21 acertos a menos), com intervalo descritivo pareado de 95% **[−0,26; +0,05] p.p.**: não houve melhoria demonstrada. A política `masked` isolada perdeu 0,31 p.p.; filtrar conflitos perdeu 0,16 p.p. O escore de `raw` desta rodada (45,26%) **não** é melhoria sobre os 45,18% históricos: o agrupamento por texto mascarado mudou os folds. Todas as políticas foram avaliadas nas mesmas novas divisões. Os intervalos não corrigem a escolha adaptativa prévia do classificador e das hipóteses.

### Situação de entrega

O pipeline clássico anterior, `artifacts/classical.joblib`, usa C=0,5. O artefato `artifacts/fixed-lexical-comparison-20260924.joblib` contém o LinearSVC experimental, que não foi promovido. O [artefato lexical de referência](research/modelo-final-logistico.md), `artifacts/final-logistic-20260928.joblib`, usa regressão logística lexical C=0,25 e foi reajustado nas 20.092 linhas originais; sua auditoria de serialização passou. O **modelo escolhido NorBERTo-512** está salvo em `artifacts/norberto-lora-final-ctx512-20260928/final_model/`; sua [avaliação no holdout](norberto/avaliacao-norberto-holdout-final.md) e a [planilha `test1` preenchida](norberto/entrega-test1-norberto-512.md) foram concluídas. **Nenhuma limpeza foi promovida** e as planilhas originais não foram alteradas. A CLI clássica foi isolada em `model_research.cli`; `src/main.py train` e `src/main.py test` são a interface pública do NorBERTo, reutilizando o treinamento final com holdout congelado e o exportador de Excel validado. As comparações repetidas no mesmo corpus são exploratórias; o gabarito externo de `test1`, se disponibilizado, será necessário para estimar generalização fora desse processo adaptativo.

## Atualização de 17/09/2026 — piloto BERTimbau

Após consolidar a MLP, foi implementado ajuste supervisionado com LoRA, pesos BERTimbau fixados por revisão, entrada sensível à caixa, truncamento início/fim e holdout interno agrupado. O primeiro piloto concluiu três épocas em 21,62 minutos no total, usando 3.514 respostas no otimizador e 2.617 na validação interna. Acurácia: BERTimbau 38,33%, baseline pareado 40,43%, lexical pareado 41,50%. O fold externo não foi avaliado. Memória CUDA máxima alocada: 696,51 MiB. Auditoria aprovada; não promover este modelo.

A curva melhorou de 35,88% para 36,72% e 38,33%, motivando um segundo piloto com taxa 0,001 em vez de 0,0002, mantendo os demais fatores. NorBERTo-base foi registrado como alternativa condicional, não testada. Protocolo, custos, métricas por classe e ressalvas sobre seleção adaptativa estão no [relatório BERTimbau](research/experimento-bertimbau.md).

O runner mais recente também salva probabilidades verificáveis por classe e tempo de inferência. Há testes de treinamento/checkpoint com BERT minúsculo em CPU, sem depender da GPU, e auditoria de resultados salvos. A advertência genérica de bootstrap foi corrigida para não chamar predições internas de OOF; o JSON bruto do primeiro piloto foi preservado e sua ressalva está documentada.

O segundo piloto terminou em 22,45 minutos. A taxa 0,001 elevou a acurácia interna para 39,82% e o F1 macro para 39,18%, ganhos de 1,49 e 1,85 p.p. sobre o primeiro piloto. Foram 39 acertos líquidos adicionais, mas o intervalo descritivo pareado do delta de acurácia foi de −0,61 a +3,67 p.p. A mudança redistribuiu as predições: recall de `c1` subiu de 25,52% para 48,61%, enquanto `c234` caiu de 33,79% para 24,94% e `c5` de 55,56% para 46,16%. O lexical pareado permaneceu superior, com 41,50%.

A auditoria do segundo piloto aprovou dados, grupos, partições, métricas e probabilidades. Split e tokenização são idênticos aos do primeiro piloto. Um blend exploratório de probabilidades encontrou complementaridade pequena, mas os pesos foram examinados no mesmo holdout e não constituem uma avaliação válida de seleção. Naquele momento, a recomendação foi testar maior volume de treino com a taxa 0,001; essa ablação foi realizada posteriormente, seguida pelos pilotos de contexto 256 e pelo NorBERTo, resumidos na seção atual.

## Diagnóstico da versão inicial

Os requisitos foram lidos em `ep1-enunciado-transcricao.md` e `ep1-modelo-relatorio-transcricao.md`.

- A métrica principal é acurácia. F1 macro e acurácia balanceada ajudam a entender erros, mas não substituem o critério do EP.
- O modelo precisa superar uma referência de regressão logística com TF-IDF. O baseline local é uma aproximação; não temos os parâmetros do baseline oficial.
- O candidato inicial, TF-IDF de palavras/caracteres com SVM, não tinha demonstrado melhora.
- A validação inicial já agrupava respostas iguais. Faltavam busca interna de parâmetros, modelos salvos, artefatos para análise dos erros, testes e dependências instaláveis.
- O README inicial apontava para arquivos e caminhos inexistentes.
- O enunciado não atribui inovação a modelos clássicos de TF-IDF. Foram adicionadas representações pré-treinadas e comparações por ablação.

## Dados

A auditoria completa está em `results/research/dataset-audit.json`. O treino contém 20.092 respostas, distribuídas entre c1 (6.347), c234 (6.853) e c5 (6.892). Há 18.143 textos distintos após normalização, 1.949 linhas repetidas adicionais e 316 grupos com rótulos conflitantes, envolvendo 1.975 linhas.

Não há textos vazios. A linha 5028 contém um valor numérico no Excel e passa a string somente para a entrada do modelo. A mediana é de 101 palavras, o percentil 95 é de 419 e o máximo é de 1.821. Os conflitos são mantidos: corrigi-los com todos os rótulos antes da validação usaria informação dos folds externos.

## Implementação e protocolo

O código foi separado em leitura/auditoria, modelos, encoders, experimentos, exportação e interface de linha de comando. Os comandos existentes de avaliação fixa e de treino seguido de predição continuam disponíveis.

Usamos três folds externos e três internos, com grupos definidos pelo SHA-256 do texto normalizado. O SHA é apenas um identificador; as relações de agrupamento são as mesmas dos textos normalizados. A alteração na ordem dos grupos muda as partições da primeira versão, por isso as comparações usam o baseline executado nos mesmos folds atuais.

A normalização decodifica entidades HTML, reduz espaços e converte para minúsculas. Mantém acentos, pontuação e negações; não usa stemming nem remoção de stopwords. Vocabulários e pesos IDF são ajustados dentro do treino de cada divisão. A busca seleciona por acurácia média interna. O resultado externo mede o pipeline selecionado, sem usar esse fold para escolher parâmetros.

As grades ficam em `configs/research/`. Os valores escolhidos e todos os scores internos ficam nos JSON de cada experimento. O campo `selected` mede a escolha de família e parâmetros realizada internamente. As outras séries são ablações para diagnóstico. A rodada de limpeza usa um executor separado e grupos definidos pelo texto mascarado, mais conservadores que os grupos de cópias exatas usados nas rodadas anteriores.

## Representações semânticas

O POTION/Model2Vec é uma alternativa estática multilíngue de 256 dimensões. Processa a resposta inteira sem truncamento e permite inferência rápida em CPU. Há uma versão isolada e outra concatenada ao TF-IDF de palavras, com peso semântico 3.

O MiniLM contextual multilíngue usa 384 dimensões e até três janelas de tokens, distribuídas do início ao fim do texto. A média dos vetores das janelas é normalizada. Textos acima desse orçamento podem ter trechos intermediários omitidos. Há uma versão isolada e outra combinada a palavras e caracteres.

Ambos usam pesos públicos em revisões fixas, sem ajuste com os rótulos do EP. Isso permite compartilhar representações congeladas entre folds sem ajustar um transformador no conjunto de validação. Os classificadores continuam sendo ajustados separadamente.

O MiniLM foi validado contra a chamada oficial `encode` para respostas curtas; a implementação também foi testada com textos longos e vazios. A NVIDIA MX550 de 2 GB foi usada nos primeiros pilotos; a AMD RX 6650 XT de 8 GB foi validada posteriormente via Docker/ROCm. Essas GPUs aceleram encoders, não o TF-IDF, a regressão logística ou o LinearSVC atuais.

## Limitações e próximos passos acadêmicos

- Pedido original e anexos não estão disponíveis. Revisão de rótulos com base apenas na resposta deve admitir casos inconclusivos; não converter automaticamente grupos conflitantes para o rótulo majoritário.
- O agrupamento original isola cópias exatas, mas respostas apenas parecidas podem atravessar folds. A rodada de limpeza usa agrupamento mais conservador pelo texto mascarado para comparações pareadas.
- A seleção de novas hipóteses após observar várias rodadas externas é adaptativa. A validação aninhada protege a escolha **dentro** de cada rodada, não transforma a sequência de pesquisas em teste independente.
- Acurácia de treino elevada não comprova generalização. Métrica principal: acurácia; F1 macro, recall por classe e diferenças pareadas ajudam a interpretar o efeito, especialmente em `c234`.
- A representação semântica é uma contribuição metodológica, mas a pontuação de inovação depende da avaliação acadêmica. Não sacrificar desempenho preditivo sem explicitar essa troca.
- A rodada de limpeza foi auditada e não demonstrou ganho; não promover seus tratamentos automaticamente. O NorBERTo final foi ajustado somente no desenvolvimento, avaliado no holdout fixo e usado para preencher `test1.xlsx`; a logística lexical de referência foi reajustada na base inteira. Permanecem pendentes o gabarito externo para medir `test1`, a identificação completa do grupo, o link do repositório, o relatório e a apresentação PDF.

## Histórico: resultado clássico consolidado

Execução completa em `results/research/classical-nested/`, semente 42, três folds externos e três internos, sem sobreposição de grupos. A busca escolheu palavras/caracteres com C=0,5 em todos os folds.

| Modelo/procedimento | Acurácia média externa | Desvio entre folds | F1 macro médio | Acurácia média de treino |
|---|---:|---:|---:|---:|
| Classe majoritária | 34,04% | 0,18 p.p. | 16,93% | 34,34% |
| Baseline: palavras + regressão logística | 43,87% | 0,24 p.p. | 43,80% | 82,75% |
| Seleção interna: palavras/caracteres + regressão logística | 44,69% | 0,20 p.p. | 44,54% | 73,94% |

O ganho médio foi de 0,82 ponto percentual, positivo nos três folds. A diferença agregada das predições OOF foi de 0,816 p.p.; o intervalo descritivo de 95% por bootstrap pareado de grupos foi de +0,289 a +1,313 p.p. Esse intervalo não incorpora toda a variabilidade do ajuste e não assegura o resultado no teste oficial.

A diferença treino/validação caiu de 38,88 para 29,25 p.p., mas continua alta. Portanto, houve melhora mensurável e redução do sobreajuste, sem evidência suficiente para afirmar que o problema de generalização foi resolvido.

## Histórico: comparações semânticas concluídas

Revisão de 16/09/2026: `static-nested`, `semantic-compact` e `semantic-nested` terminaram, com três folds externos e três internos, semente 42 e todas as combinações das respectivas grades. Os hashes da base, as divisões em folds e as 20.092 linhas OOF coincidem com a execução clássica. As acurácias agregadas foram recalculadas a partir das predições e conferem com os relatórios.

| Execução | Acurácia de `selected` | Acurácia da ablação híbrida | Acurácia dos embeddings isolados |
|---|---:|---:|---:|
| POTION/Model2Vec | 43,83% | 44,01% | 41,78% |
| MiniLM — grade compacta | 44,02% | 44,43% | 42,58% |
| MiniLM — grade completa | 44,62% | 44,62% | 42,58% |

O procedimento de seleção pode escolher famílias diferentes em cada fold. No POTION, escolheu baseline/híbrido/baseline; no MiniLM compacto, baseline/híbrido/híbrido. Na grade contextual completa, escolheu o híbrido em todos os folds, sempre com C=0,5 e pesos iguais a 1 para palavras, caracteres e semântica.

O híbrido completo superou o baseline em aproximadamente 0,75 p.p., mas não demonstrou vantagem sobre o clássico de 44,69%. O delta agregado clássico menos híbrido é de 0,065 p.p.; seu intervalo descritivo pareado de 95% é de −0,344 a +0,458 p.p. As duas abordagens estão próximas dentro dessa incerteza. Não há justificativa para promover os embeddings isolados nesta rodada.

Naquela etapa, o modelo clássico foi treinado em toda a base e salvo em `artifacts/classical.joblib`, com metadados em `artifacts/classical.json`. A seleção daquele experimento escolheu palavras/caracteres e C=0,5. Ainda não havia artefato semântico ajustado para entrega nem planilha de teste oficial rotulada; o NorBERTo-512 foi salvo posteriormente.

## Histórico: LinearSVC e regularização

A configuração `configs/research/linear-svc.json` compara quatro famílias nos mesmos folds:

| Família | Representação | Valores de C |
|---|---|---|
| Baseline | Palavras, regressão logística | 2, fixo |
| Referência lexical | Palavras/caracteres, regressão logística | 0,1; 0,25; 0,5 |
| SVM de palavras | Mesmo TF-IDF de palavras do baseline | 0,01; 0,05; 0,1; 0,25; 0,5; 1 |
| SVM lexical | Mesmo TF-IDF de palavras/caracteres da referência lexical | 0,01; 0,05; 0,1; 0,25; 0,5; 1 |

A grade amplia a regularização porque C=0,5 foi o menor valor da grade lexical anterior e venceu em todos os folds. Os SVMs usam `LinearSVC`, `dual="auto"`, `max_iter=10000`, `tol=1e-4` e pesos de classe uniformes. O conjunto é aproximadamente balanceado; pesos de classe e outras sementes podem ser avaliados em uma rodada posterior.

A seleção continua usando somente a acurácia interna, com vocabulário/IDF ajustados dentro de cada divisão. A rodada é exploratória, orientada pelos resultados anteriores. SVM com TF-IDF continua sendo um método clássico para o critério de inovação do EP.

A execução de 16/09/2026 terminou em 1.841,69 s (30,69 min), com todos os 144 ajustes internos, 12 reajustes das famílias e três referências majoritárias. A conferência dos arquivos verificou configuração, hashes, predições, métricas e as mesmas divisões externas das quatro rodadas anteriores. O log não contém avisos de convergência nem falhas de ajuste.

| Modelo/procedimento | Acurácia média externa | F1 macro médio | C escolhido nos folds 1/2/3 |
|---|---:|---:|---|
| Baseline | 43,87% | 43,80% | 2 / 2 / 2 |
| Logística palavras/caracteres | 45,14% | 44,91% | 0,25 / 0,25 / 0,25 |
| LinearSVC palavras | 44,91% | 44,52% | 0,05 / 0,1 / 0,1 |
| LinearSVC palavras/caracteres | 45,19% | 44,87% | 0,05 / 0,05 / 0,05 |
| Seleção interna (`selected`) | 45,02% | 44,69% | SVM palavras / logística / logística |

O SVM palavras/caracteres acertou apenas nove linhas a mais que a logística regularizada: delta OOF de +0,045 p.p., intervalo descritivo de −0,181 a +0,260 p.p. A melhora da logística frente ao seu resultado anterior foi de +0,453 p.p., intervalo de +0,116 a +0,802 p.p. Isso favorece a hipótese de que regularizar melhor foi mais importante que trocar o classificador.

O novo procedimento completo (`selected`) ganhou +1,145 p.p. OOF sobre o baseline, mas apenas +0,328 p.p. sobre o procedimento clássico anterior; este último intervalo inclui zero (−0,060 a +0,770 p.p.). Não declarar superioridade definitiva sobre a rodada anterior nem usar 45,19% como score de `selected`.

As buscas de SVM palavras/caracteres consumiram 13,89 min, contra 2,38 min do SVM de palavras com o mesmo número de valores de C. Não se justifica repetir a grade inteira: os melhores C não estão mais nas bordas. A análise completa, incluindo custos, limites e critérios definidos naquela etapa, está em [avaliacao-linear-svc.md](research/avaliacao-linear-svc.md). Nenhum novo experimento havia sido iniciado **na revisão de 16/09**; as rodadas posteriores estão resumidas no início deste registro.

## Histórico: MiniLM congelado + MLP pequena

Após a análise do LinearSVC, a prioridade foi alterada para uma família não linear, em vez de ampliar a busca dos modelos atuais. Foi implementado `semantic_mlp`, com os mesmos embeddings MiniLM de 384 dimensões e uma camada oculta ReLU de 64 neurônios. A grade `configs/research/semantic-mlp.json` testa apenas `alpha` 0,1 e 1,0. Mantém baseline, logística lexical C=0,25 e controle semântico linear C em {1, 10, 100}.

O encoder permanece congelado e seu cache não depende dos rótulos. Nenhum escalonamento adicional ou alteração do pooling foi introduzido: o objetivo é isolar o classificador. A divisão aleatória usada por `MLPClassifier(early_stopping=True)` não é empregada; a parada depende apenas da perda de treino ou do máximo de 200 épocas. Os dois níveis de validação continuam agrupados.

Novos relatórios passam a incluir tempos médios/desvios de ajuste e scoring por configuração, tempo de reajuste e diagnósticos do classificador vencedor (iterações e, para MLP, perda final e curva de perda de treino). Esses diagnósticos se referem ao reajuste, não a todos os ajustes internos; avisos de convergência são contados por busca e continuam visíveis no log. Resultados antigos não foram reescritos.

Testes cobrem arquitetura, semente, clonagem, identidade da representação com o controle, seleção aninhada, cache compartilhado sem recodificação, persistência e predição sem alteração dos pesos. O protocolo e o registro do piloto estão em [experimento-semantic-mlp.md](research/experimento-semantic-mlp.md). Ainda não há modelo final MLP treinado para entrega.

Consolidação de 17/09/2026: execução completa em 22,89 min, com 63 ajustes internos e 20.092 predições OOF conferidas. A MLP atingiu 43,68% de acurácia média, frente a 42,58% do controle semântico linear, 43,87% do baseline e 45,14% da logística lexical. `selected` escolheu a logística lexical em todos os folds. Os controles reproduziram as predições anteriores; nenhum novo treinamento foi necessário para conferir as métricas.

O ganho OOF MLP menos semântico linear foi de +1,115 p.p., intervalo descritivo pareado de 95% [+0,425; +1,792]. Frente à logística lexical, a diferença foi −1,448 p.p., intervalo [−2,258; −0,683]. Alpha=0,1 venceu nos três folds; os reajustes executaram 200/200/131 épocas e houve dez avisos de limite de iterações no conjunto das buscas MLP. A família consumiu 18,88 min (82,5% do total). Esses dados mostram ganho sobre o controle semântico, mas não justificam promover a MLP ou presumir que mais épocas resolverão a diferença.

A decisão tomada após a MLP foi testar BERTimbau Base supervisionado na NVIDIA MX550 e, posteriormente, na AMD RX 6650 XT. Essas execuções e a comparação com NorBERTo foram concluídas; os resultados estão na seção atual e no relatório de ajuste supervisionado. A decisão histórica não altera retrospectivamente os resultados da MLP.
