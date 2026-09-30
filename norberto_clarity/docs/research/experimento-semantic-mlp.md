# Experimento: MiniLM congelado + MLP pequena

Estado: **concluído**, revisão de 17/09/2026. Execução entre 16/09 às 23h44 e 17/09 às 00h07 (America/Sao_Paulo). Artefatos completos em [results/research/semantic-mlp-nested](../../results/research/semantic-mlp-nested/results.json).

## Resultado consolidado

A MLP melhorou sobre o controle semântico linear, mas não superou o baseline nem a referência lexical. Não foi promovida a modelo final. Os 45,14% de `selected` pertencem à logística lexical, escolhida internamente nos três folds, e não à MLP.

| Modelo/procedimento | Acurácia média externa ± desvio | F1 macro médio | Acurácia média de treino |
|---|---:|---:|---:|
| Baseline | 43,87% ± 0,24 p.p. | 43,80% | 82,75% |
| Logística palavras/caracteres | 45,14% ± 0,55 p.p. | 44,91% | 67,06% |
| MiniLM + logística | 42,58% ± 0,40 p.p. | 42,31% | 46,32% |
| MiniLM + MLP | 43,68% ± 0,60 p.p. | 43,56% | 52,35% |
| `selected` (logística lexical) | 45,14% ± 0,55 p.p. | 44,91% | 67,06% |

Conferidos os 63 ajustes internos, os três folds parciais, o manifesto, o hash da base e do código e as 20.092 predições OOF. As métricas recalculadas coincidem com os JSON. Não há grupos compartilhados entre treino e validação nas divisões externas/internas reconstruídas. Os controles reproduzem as predições dos experimentos anteriores nos mesmos folds. Isso não elimina a limitação de duplicatas aproximadas.

### Comparações pareadas e erros

Bootstrap pareado de grupos normalizados, 2.000 reamostragens, semente 42, usando `model_research.experiment.paired_group_bootstrap`. Diferenças abaixo usam acurácia OOF agregada, não a média não ponderada dos folds da tabela anterior.

| Comparação | Delta OOF | Intervalo descritivo de 95% | Saldo de acertos |
|---|---:|---:|---:|
| MLP − MiniLM + logística | +1,115 p.p. | [+0,425; +1,792] p.p. | +224 |
| MLP − logística lexical | −1,448 p.p. | [−2,258; −0,683] p.p. | −291 |
| MLP − baseline | −0,179 p.p. | [−1,037; +0,605] p.p. | −36 |

Os intervalos são condicionais às predições observadas; não repetem treinamento, não corrigem comparações múltiplas nem a adaptação do projeto após rodadas anteriores. Não constituem garantia para o teste oficial.

| Modelo | Recall c1 | Recall c234 | Recall c5 |
|---|---:|---:|---:|
| Logística lexical | 44,02% | 36,45% | 54,80% |
| MiniLM + logística | 42,41% | 34,07% | 51,18% |
| MiniLM + MLP | 49,14% | 35,23% | 47,08% |

A MLP melhora a identificação de c1, mas perde principalmente em c5 frente à referência lexical. Também reduz o recall c234 em 1,23 p.p. frente a essa referência. Não atingiu o filtro prático definido antes da rodada: faltou ganho de acurácia e a queda de recall c234 excedeu 1 p.p.

### Seleção, convergência e custo

| Fold | Alpha MLP escolhido | Acurácia externa MLP | Acurácia externa lexical | Épocas do reajuste MLP | Avisos na busca MLP |
|---|---:|---:|---:|---:|---:|
| 1 | 0,1 | 42,84% | 45,91% | 200 | 4 |
| 2 | 0,1 | 44,13% | 44,86% | 200 | 4 |
| 3 | 0,1 | 44,08% | 44,65% | 131 | 2 |

Houve dez avisos de limite de 200 épocas. São contagens agregadas de ajustes internos e reajuste de cada busca, não dez folds nem dez reajustes. Os reajustes dos folds 1/2 atingiram o limite; o terceiro parou pelo critério de perda de treino. As perdas finais foram 1,02397, 1,02393 e 1,05338. As curvas estão em `refit_diagnostics.training_loss_curve`.

Alpha=0,1 venceu na borda inferior da grade, e houve treinamento limitado por épocas. Portanto, a família não está esgotada. Porém, isso não demonstra que mais épocas ou menor regularização superarão o lexical. O gap médio treino/validação da MLP foi 8,66 p.p.; a acurácia de treino dos reajustes variou de 45,90% a 55,92%, sem que possamos isolar efeito de partição e inicialização nesta única semente externa.

Tempo registrado: **1.373,56 s = 22,89 min**, abaixo da projeção inicial de 25–35 min. Esse cronômetro termina antes dos bootstraps e da escrita final, não mede todo o tempo do processo.

| Componente | Tempo |
|---|---:|
| Busca baseline, com reajustes | 0,62 min |
| Busca logística lexical, com reajustes | 1,54 min |
| Busca MiniLM + logística, com reajustes | 0,78 min |
| Busca MiniLM + MLP, com reajustes | 18,88 min |
| Demais operações, por diferença | 1,07 min |

A MLP consumiu 82,5% do tempo. Esses valores incluem transformações, cache, scoring e reajustes; não são benchmarks isolados de inferência nem comparações de grades do mesmo tamanho.

### Decisão após esta rodada

Manter a logística lexical como referência forte e não ampliar automaticamente a MLP. O próximo experimento é o ajuste supervisionado de **BERTimbau Base (`neuralmind/bert-base-portuguese-cased`)**, começando na **NVIDIA MX550 de 2 GB**. A AMD RX 6650 XT de 8 GB em Linux fica como alternativa posterior, caso o experimento seja promissor e o ambiente seja validado.

Primeiro medir memória e custo de treinamento; depois executar a avaliação dentro do orçamento observado. Não presumir que o treinamento caiba em 2 GB nem reaproveitar embeddings congelados como se fossem representações de um encoder ajustado. Preservar a validação por grupos, separar seleção interna e avaliação externa e registrar qualquer adaptação de protocolo como exploratória. Nenhum resultado do BERTimbau está estabelecido nesta consolidação.

## Pergunta e controles

Hipótese: uma função não linear consegue extrair sinal dos embeddings MiniLM que a regressão logística não aproveita? Isso não pressupõe que os embeddings contenham informação suficiente para superar o modelo lexical.

O controle MiniLM + logística obteve anteriormente 42,58%; a logística palavras/caracteres regularizada chegou a 45,14%. Melhorar somente sobre o primeiro não basta para substituir o melhor modelo disponível. A rodada LinearSVC não demonstrou vantagem clara sobre a logística lexical.

Configuração: [semantic-mlp.json](../../configs/research/semantic-mlp.json).

| Candidato | Representação | Parâmetros buscados |
|---|---|---|
| `baseline` | TF-IDF palavras + logística | C=2, fixo |
| `word_char_lr` | TF-IDF palavras/caracteres + logística | C=0,25, fixo |
| `semantic_lr` | MiniLM congelado + logística | C em {1; 10; 100} |
| `semantic_mlp` | Mesmo MiniLM congelado + MLP | alpha em {0,1; 1,0} |

A grade de C do controle semântico é preservada para evitar comparar a MLP com uma referência arbitrariamente enfraquecida. A referência lexical é fixa nesta rodada porque C=0,25 foi escolhido na investigação anterior; portanto a nova rodada também é exploratória.

## Arquitetura e limites

- Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisão fixa já usada no projeto. Nenhum ajuste supervisionado do encoder.
- Representação: 384 dimensões, até três janelas por texto, média e normalização L2; exatamente o mesmo `FrozenEmbeddings` do controle linear. Sem novo scaler ou redução dimensional.
- Classificador: `MLPClassifier`, uma camada oculta com 64 neurônios ReLU e saída multiclasse. São 24.835 parâmetros treináveis com entrada de 384 dimensões e três classes.
- Adam, taxa inicial 0,001, lotes de 128, embaralhamento com semente controlada; até 200 épocas. Regularização L2 com alpha 0,1 ou 1,0. Alpha não é o mesmo parâmetro C dos modelos anteriores.
- `early_stopping=False`: não usa o holdout aleatório interno do scikit-learn. A condição de parada monitora apenas a perda de treino com tolerância 0,0001 e paciência de 15 épocas, além do limite de 200 épocas. Não se trata de early stopping por generalização.
- O máximo de épocas limita o custo, mas não garante convergência. Se houver `ConvergenceWarning`, examinar as curvas e os avisos antes de atribuir eventual resultado ruim à família MLP. Não aumentar épocas olhando scores externos durante a execução.
- CPU, dois threads, uma busca por vez. O scikit-learn não usa a GPU para esta MLP; com embeddings já armazenados, não há necessidade de recodificação CUDA.

## Validação e custo

Manter três folds externos e três internos, com grupos de textos normalizados. Semente externa 42 e sementes internas 43/44/45. Inicializações e embaralhamento da MLP seguem a semente de cada busca. O primeiro teste não mede variabilidade entre várias inicializações; isso só será investigado se a família se mostrar promissora.

Sete configurações × três folds internos × três externos = **63 ajustes internos**, mais 12 reajustes vencedores das quatro famílias e três ajustes da referência majoritária. `selected` reutiliza as predições do vencedor interno. Os controles são executados nos mesmos folds; resultados históricos não são misturados como se pertencessem a esta seleção.

Antes da execução completa, o piloto usa o primeiro treino interno do primeiro fold externo, com os dois valores de alpha e o limite real de épocas. Ele verifica cobertura do cache em toda a base e mede ajuste/predição sem calcular scores para selecionar configurações. Não modifica a grade com base em acurácia de piloto. O processo do piloto tem limite de cinco minutos.

O registro local do piloto é `artifacts/semantic-mlp-pilot.json`. Ele contém tamanhos das partições, parâmetros, tempos, número de épocas, avisos, hash dos dados e código. O cache é verificado impedindo o carregamento do encoder: qualquer texto ausente encerra o piloto antes de recodificar. Modo offline na execução completa impede downloads, mas não é uma garantia de cache completo; essa garantia vem da verificação prévia na base atual.

### Piloto concluído em 16/09/2026

Cache confirmado para as 20.092 linhas (384 dimensões), leitura em 0,94 s. Partição representativa: 9.065 linhas de treino e 4.414 de validação; dois threads.

| Alpha | Ajuste | Predição | Épocas | Aviso |
|---|---:|---:|---:|---|
| 0,1 | 86,28 s | 0,35 s | 200 | Limite de épocas sem convergência declarada |
| 1,0 | 9,19 s | 0,39 s | 39 | Nenhum |

A grade foi mantida sem consultar acurácia de piloto. O aviso de alpha=0,1 é uma limitação conhecida do orçamento, não um erro de execução nem uma convergência bem-sucedida. A MLP continua sendo uma hipótese a avaliar com cautela.

Projeção de **25–35 minutos** para a rodada completa: aproximadamente 14,3 minutos para os 18 ajustes internos MLP extrapolados do piloto, até cerca de 6,5 minutos para seus três reajustes se vencer a configuração longa, mais controles e operações externas. Tempos podem variar; a estimativa não é um limite automático de execução. O limite de cinco minutos aplica-se apenas ao processo de piloto. A busca completa é limitada pela grade e pelo máximo de épocas, sem agendamento automático de novas rodadas.

Novos campos do relatório:

- Por configuração: `mean_fit_seconds`, `std_fit_seconds`, `mean_score_seconds`, `std_score_seconds`.
- Por busca: `refit_seconds` e `refit_diagnostics` do vencedor reajustado, com `n_iter`, `training_loss` e `training_loss_curve` quando suportados.
- Por busca: `convergence_warnings`, com mensagens e contagens de avisos nos ajustes internos e no reajuste. As contagens são agregadas por família, não atribuídas a cada configuração; os avisos também continuam no log.
- `seconds` continua medindo a busca inteira da família. Fit inclui transformação/carregamento de cache, não apenas o otimizador da MLP. Scoring inclui transformação e predição.

As curvas persistidas são dos reajustes vencedores, não das redes descartadas nem de todos os folds internos. O log e as contagens de avisos complementam esses diagnósticos; ausência de um aviso não garante boa generalização.

## Reprodução da execução concluída

```bash
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src .venv/bin/python -u -m model_research.cli experiment \
  --train data/train.xlsx --config configs/research/semantic-mlp.json \
  --folds 3 --inner-folds 3 --seed 42 --threads 2 --verbose 2 \
  --output-dir results/research/semantic-mlp-reproducao
```

Resultados originais, agora em `results/research/semantic-mlp-nested/`. Log local: `artifacts/logs/semantic-mlp.log`. O comando acima usa outro diretório para não colidir com os resultados concluídos.

- `manifest.json` é escrito no início e registra configuração, código e dados.
- `fold-1.json`, `fold-2.json` e `fold-3.json` preservam cada fold concluído.
- `results.json` e `oof.csv` são escritos ao terminar a avaliação completa.
- Use outro diretório para repetir. O comando não sobrescreve uma execução anterior nem retoma automaticamente folds parciais.
- O comando `experiment` não salva modelo final para entrega; essa etapa continua separada em `train`.

## Critérios definidos antes da execução

1. Conferir completude, integridade das predições, ausência de sobreposição de grupos e avisos de otimização.
2. Comparar MLP com logística semântica para isolar o classificador e com logística lexical para medir utilidade real. Examinar a acurácia média externa, F1 macro, recall c234, diferença treino/validação e deltas OOF pareados por grupos.
3. Separar a métrica de família MLP da métrica `selected`; o controle pode vencer a seleção interna em parte ou em todos os folds.
4. Considerar promissor ganho médio de pelo menos 0,3 p.p. sobre a logística lexical, positivo em pelo menos dois folds, sem queda maior que 1 p.p. no recall c234. É um filtro prático de investigação, não significância estatística nem garantia de teste.
5. Se superar somente o controle semântico, registrar melhora de representação/classificação sem promover a MLP a modelo final. Se a otimização tiver estabilizado e a família não melhorar, não ampliar automaticamente arquitetura, alphas ou épocas.
6. Somente uma hipótese promissora justifica repetir uma semente ou investigar ajuste supervisionado do encoder. Nenhuma segunda rodada está agendada automaticamente.

Esta rodada não resolve as limitações de rótulos conflitantes, duplicatas aproximadas e possível falta de contexto da requisição. Maior capacidade também pode aumentar sobreajuste.
