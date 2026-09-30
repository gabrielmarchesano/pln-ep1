# Avaliação LinearSVC e planejamento de experimentos

**Registro histórico da etapa lexical.** A decisão vigente sobre o modelo de entrega é o [NorBERTo-512 LoRA](../modelo-escolhido-norberto-512.md); o SVM não foi promovido e a regressão logística permaneceu como referência.

Revisão: 16/09/2026. Resultados: [linear-svc-nested](../../results/research/linear-svc-nested/results.json). Configuração executada: [linear-svc.json](../../configs/research/linear-svc.json).

Atualização de prioridade: após discutir o custo de continuar ajustando modelos lineares, foi escolhida a rodada **MiniLM congelado + MLP pequena**, descrita em [experimento-semantic-mlp.md](experimento-semantic-mlp.md). O planejamento abaixo preserva o contexto da avaliação LinearSVC; a grade híbrida de regularização deixou de ser a próxima execução. As métricas e conclusões históricas não mudaram.

## Conclusão e decisão daquela etapa

A busca terminou corretamente. O ganho mais convincente desta rodada veio de aumentar a regularização da regressão logística. LinearSVC com palavras/caracteres teve a maior acurácia média de família, mas ficou praticamente empatado com a logística regularizada. Não há motivo para repetir a grade ampla nem para promover automaticamente o SVM a modelo final.

Manter a logística palavras/caracteres com C=0,25 como referência forte da próxima investigação e SVM de palavras como alternativa de menor custo. Conservar o SVM palavras/caracteres na comparação curta, sem nova busca extensa. São decisões exploratórias, não resultados de teste independente.

Nenhum novo experimento foi iniciado nesta revisão. `artifacts/classical.joblib` ainda corresponde à rodada anterior, C=0,5. Os resultados novos não são um modelo final salvo. O teste oficial continua ausente.

## Integridade e protocolo

- Treino: 20.092 linhas, 18.143 grupos de textos normalizados; SHA-256 `0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`.
- Validação aninhada, três folds externos e três internos; semente externa 42, sementes internas 43/44/45. Vetorizadores aprendidos apenas dentro de cada treino.
- 16 configurações × 3 folds internos × 3 externos = 144 ajustes internos; mais 12 reajustes vencedores das famílias e três ajustes majoritários. `selected` reutiliza o vencedor, sem ajuste adicional.
- Conferidos `manifest.json`, os três `fold-N.json`, `results.json` e `oof.csv`; configuração e hashes do código coincidem com a implementação executada.
- Cada linha tem uma predição por família, sem valores ausentes. Cada grupo aparece em apenas um fold externo. As partições, rótulos e predições do baseline coincidem com as quatro rodadas anteriores.
- Acurácias agregadas e por fold, F1 macro agregado, acurácia balanceada agregada e matrizes de confusão foram recalculados a partir de OOF. `selected` coincide com o vencedor da busca interna em cada fold.
- Todas as combinações previstas estão registradas, com três scores internos cada. O log local `artifacts/logs/linear-svc.log` não contém `ConvergenceWarning`, `Traceback` ou falha de ajuste.

Não foi necessário repetir treinamento para fazer essas verificações. Esta revisão não altera o código de modelagem.

## Desempenho

A métrica principal é a média das acurácias dos folds externos; o desvio abaixo é entre folds, não um intervalo de confiança. Acurácia OOF agrega todas as linhas e difere ligeiramente porque os folds têm tamanhos diferentes.

| Família/procedimento | Acurácia média ± desvio | Acurácia OOF | F1 macro médio | Acurácia de treino média |
|---|---:|---:|---:|---:|
| Baseline fixo | 43,87% ± 0,24 p.p. | 43,87% | 43,80% | 82,75% |
| Logística palavras/caracteres | 45,14% ± 0,55 p.p. | 45,14% | 44,91% | 67,06% |
| SVM palavras | 44,91% ± 0,52 p.p. | 44,91% | 44,52% | 69,63% |
| SVM palavras/caracteres | 45,19% ± 0,54 p.p. | 45,18% | 44,87% | 68,90% |
| `selected` | 45,02% ± 0,37 p.p. | 45,01% | 44,69% | 66,60% |
| Classe majoritária | 34,04% ± 0,18 p.p. | 34,04% | 16,93% | 34,34% |

| Fold | C logística lexical | C SVM palavras | C SVM lexical | Vencedor interno | Acurácia externa de `selected` |
|---|---:|---:|---:|---|---:|
| 1 | 0,25 | 0,05 | 0,05 | SVM palavras | 45,53% |
| 2 | 0,25 | 0,1 | 0,05 | Logística lexical | 44,86% |
| 3 | 0,25 | 0,1 | 0,05 | Logística lexical | 44,65% |

O SVM lexical não foi escolhido em nenhum fold interno. Não há contradição: a seleção não enxerga os rótulos externos e não precisa escolher o maior score externo. Reportar 45,19% como desempenho do procedimento de seleção seria incorreto.

Os melhores C ficaram no interior das grades. C=0,25 da logística venceu nos três folds; C=0,05 do SVM lexical também. Nos SVMs, C≥0,25 foi pior internamente que o vencedor em todos os folds. Não há evidência para ampliar a grade nessa direção. A diferença treino/validação da logística caiu de 29,25 p.p. na rodada anterior para 21,92 p.p.; ainda há generalização limitada.

### Diferenças pareadas

Bootstrap pareado de grupos normalizados, 2.000 reamostragens, semente 42, usando `model_research.experiment.paired_group_bootstrap`. Diferenças calculadas na acurácia OOF, em pontos percentuais.

| Comparação (primeiro menos segundo) | Delta | Intervalo descritivo de 95% |
|---|---:|---:|
| Novo `selected` − baseline | +1,145 | [+0,570; +1,686] |
| Novo `selected` − clássico anterior | +0,328 | [−0,060; +0,770] |
| Novo `selected` − semântico completo | +0,393 | [−0,084; +0,915] |
| Nova logística − logística anterior | +0,453 | [+0,116; +0,802] |
| SVM lexical − nova logística | +0,045 | [−0,181; +0,260] |
| SVM lexical − SVM palavras | +0,269 | [−0,174; +0,711] |

O SVM lexical corrige 257 erros da nova logística, mas perde 248 acertos dela: saldo de nove linhas. Os caracteres também não demonstraram vantagem clara sobre SVM de palavras dentro dessa incerteza. Isso não prova equivalência; significa que estes dados não sustentam uma preferência forte por acurácia.

Esses intervalos condicionam-se às predições já obtidas: não repetem o treinamento, não corrigem múltiplas comparações nem a adaptação das grades após observar resultados. A seleção aninhada protege cada busca interna, mas as rodadas sucessivas continuam exploratórias. Novas sementes medem sensibilidade às partições, não criam um teste independente. A justificativa de manter seleção e avaliação separadas está no [exemplo oficial de validação aninhada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### Onde ainda erramos

| Modelo | Recall c1 | Recall c234 | Recall c5 |
|---|---:|---:|---:|
| Baseline | 43,67% | 37,36% | 50,52% |
| Nova logística lexical | 44,02% | 36,45% | 54,80% |
| SVM palavras | 43,50% | 34,55% | 56,52% |
| SVM lexical | 45,08% | 34,79% | 55,62% |

O avanço em acurácia não é uniforme: os modelos melhoram principalmente c5, enquanto o recall de c234 cai. No SVM lexical, das 6.853 respostas c234, 2.498 viram c5, 1.971 viram c1 e apenas 2.384 são acertadas. A logística tem F1 macro ligeiramente melhor que o SVM lexical, embora a métrica principal continue sendo acurácia.

Priorizar análise qualitativa de c234 e das discordâncias entre modelos. Há 316 grupos com rótulos conflitantes no corpus; não corrigir/remover esses exemplos usando os rótulos de toda a base antes da validação. Não atribuir todo erro a ruído de anotação sem examinar o texto e o contexto disponível.

## Custo observado e limites da medição

Tempo registrado pelo experimento: **1.841,69 s = 30,69 min**, CPU, dois threads e `n_jobs=1`. O cronômetro termina antes dos bootstraps e da escrita final de OOF/JSON; não é o tempo completo do processo.

| Componente | Ajustes internos | Tempo acumulado | Fração do total |
|---|---:|---:|---:|
| Busca baseline, incluindo reajustes | 9 | 1,20 min | 3,9% |
| Busca logística lexical, incluindo reajustes | 27 | 8,85 min | 28,8% |
| Busca SVM palavras, incluindo reajustes | 54 | 2,38 min | 7,8% |
| Busca SVM lexical, incluindo reajustes | 54 | 13,89 min | 45,3% |
| Demais operações, por diferença | — | 4,38 min | 14,3% |

As duas buscas SVM têm o mesmo tamanho; a lexical custou aproximadamente 5,84 vezes a de palavras nesta execução. Esse é custo de busca, com vetorização, acesso ao cache, ajuste, scoring e reajuste; não é comparação isolada dos solvers nem benchmark de inferência. A ordem das famílias e caches aquecidos também afetam o resultado. O restante inclui predições externas e de treino, métricas e outras operações, não apenas I/O.

Antes de otimizar, registrar `mean_fit_time`, `std_fit_time`, `mean_score_time`, `std_score_time` e `refit_time_`, já fornecidos por [GridSearchCV 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html). Esses campos não foram persistidos nesta rodada; não é possível decompor retroativamente seu tempo com precisão.

O cache atual de `Pipeline` reaproveita o ajuste/transformação de treino, mas `predict` volta a transformar os textos. Isso foi conferido na implementação instalada do scikit-learn 1.7.2 e é consistente com a [documentação de Pipeline](https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html). Há uma oportunidade de compartilhar matrizes de validação entre valores de C e classificadores com representação idêntica. É uma hipótese de otimização, não um ganho medido.

Se implementado, o cache deve identificar base, índices de treino/validação, parâmetros do vetorizador, normalização e versão de código. Nunca ajustar TF-IDF globalmente para ganhar velocidade. Antes de adotar outro executor, testar paridade de predições, seleção, desempates e ausência de vazamento contra o pipeline atual em uma amostra agrupada. Manter matrizes esparsas e avaliar RAM/disco; não ativar paralelismo amplo sem medir.

## Plano proposto, com orçamento e critérios de parada

As etapas abaixo são propostas, não execuções já realizadas nem configurações novas implementadas. Tempos são estimativas para este computador, com dois threads e caches existentes; não são garantias. Confirmar a estimativa após o primeiro fold. Usar diretórios novos e manter os experimentos anteriores imutáveis.

### 1. Diagnóstico barato e instrumentação antes de nova busca

- Sem novo ajuste: separar erros OOF por classe, comprimento, grupos conflitantes/não conflitantes e concordância entre modelos. Revisar cerca de 60 textos, amostrados por esses estratos; registrar hipóteses antes de testá-las. Não publicar textos potencialmente pessoais nos relatórios.
- Acrescentar a medição detalhada acima e fazer um piloto de no máximo **5 minutos**, em uma amostra de grupos, para distinguir custo de transformação e ajuste. Piloto serve para custo/correção, não para escolher o vencedor pelo score de uma amostra pequena.
- Implementar cache adicional somente se o perfil indicar economia relevante e os testes de paridade passarem. Não refatorar toda a busca apenas com base no tempo total.

### 2. Próxima rodada de modelo: regularização do híbrido já disponível

Esta é a prioridade para investigar ganho e uma abordagem além de TF-IDF: a grade semântica anterior também parou em C=0,5, agora sabemos que o controle lexical melhorou com C=0,25.

- Famílias: baseline fixo; logística lexical fixa C=0,25; híbrido MiniLM com C em `{0,1; 0,25; 0,5}`, pesos palavra/caractere/semântica todos iguais a 1.
- Cinco configurações, **45 ajustes internos + 9 reajustes**, mais três referências majoritárias; manter 3×3 folds e semente 42 para comparação exploratória pareada. Não recalcular embeddings nem repetir a família semântica isolada, já inferior nesta investigação.
- Estimativa inicial **12–20 minutos**, baseada na redução da grade contextual anterior de dez para cinco configurações e nos custos lexicais observados. Há custos fixos e famílias diferentes; não assumir redução linear exata.
- Orçamento proposto: revisar a projeção após o primeiro fold; se apontar mais de **25 minutos**, parar para replanejar e não publicar resultado parcial como avaliação completa. O executor atual não tem limite automático nem retomada de folds: essa revisão precisa de acompanhamento, e interrupção pode exigir reinício em outro diretório.
- Critério prático prévio: considerar promissor ganho médio de pelo menos **0,3 p.p.** sobre a logística controle, positivo em pelo menos dois folds, sem piora maior que 1 p.p. de recall c234. É um filtro operacional, não teste de significância. Examinar também OOF pareado; resultado abaixo do filtro não justifica ampliar pesos/encoders automaticamente.
- Se falhar, documentar a ablação e encerrar essa direção por enquanto. Não tentar compensar com uma grade combinatória maior.

### 3. Verificar estabilidade com uma lista curta

Se a prioridade for concluir rapidamente, pular a etapa 2 e ir diretamente para esta. Fixar a lista antes de rodar novas sementes.

- Lista clássica: baseline; logística lexical C=0,25; SVM palavras C em `{0,05; 0,1}`; SVM lexical C=0,05.
- Cinco configurações, **45 ajustes internos + 12 reajustes**, mais três referências majoritárias, contra 144 ajustes internos na rodada concluída: redução de 68,75% nas avaliações internas, não necessariamente no tempo total.
- Executar primeiro semente **17**; estimativa **12–18 minutos**, com revisão de orçamento de **20 minutos** após o primeiro fold. Se o híbrido passar a etapa 2, incluir apenas o seu C escolhido para desenvolvimento: seis configurações, 54 ajustes internos + 15 reajustes; revisar orçamento para aproximadamente **15–25 minutos**.
- Reservar semente **137** apenas se a incerteza puder mudar a escolha final e houver orçamento. Não repetir a busca ampla na semente 42 nem escolher apenas a semente mais favorável. Quando executadas, apresentar todas as sementes e diferenças pareadas dentro de cada uma; não tratar seus folds como observações independentes.
- Se diferenças permanecerem menores que o limiar prático de 0,3 p.p. e instáveis, preferir o candidato de menor custo medido no uso pretendido, registrando que isso é uma decisão operacional. Avaliar separadamente custo de busca, treinamento final e predição; o menor custo de busca não prova menor latência de produção.

### 4. Somente depois: uma hipótese nova por vez

Se a leitura dos erros justificar, testar poucas características explícitas de forma/clareza (comprimento, proporção de dígitos, links, frases e referências a anexos) junto ao melhor controle. Estabelecer um único conjunto de características antes da rodada, ajustar qualquer normalização somente no treino e fazer ablação com/sem características. Isso requer implementação; não foi avaliado nem tem ganho assegurado. A contribuição acadêmica precisa ser justificada, não presumida.

Não priorizar agora fine-tuning de transformer, novos encoders, SVM RBF, grades de pesos de classe ou ensembles extensos. A base é aproximadamente balanceada; o recall baixo de c234 não implica automaticamente que `class_weight="balanced"` resolva. A MX550 tem apenas 2 GB de VRAM, e o executor atual de LinearSVC usa CPU. CUDA pode ajudar encoders, não acelera esta busca LinearSVC tal como implementada.

### 5. Encerrar e preparar a entrega

Congelar protocolo e configuração, executar a seleção interna final em toda a base uma única vez e salvar pipeline/metadados. A grade final pode ser diferente da ampla original, mas deve ser documentada como procedimento novo; não herda automaticamente o score `selected` antigo. O score da seleção final não é avaliação de teste.

Reservar tempo para o relatório acadêmico, apresentação PDF e Excel oficial, ainda pendentes. SVM/TF-IDF não pontua sozinho em inovação segundo o enunciado; o híbrido deve ser discutido mesmo que o clássico continue sendo a melhor escolha preditiva. Não sacrificar desempenho sem explicitar a troca, nem prometer pontuação de inovação.

## Reprodução das comparações adicionais, sem treinar

Executar na raiz, com o ambiente instalado. As comparações contra baseline já estão no JSON original; este trecho reproduz as cinco comparações adicionais entre rodadas/famílias, usando os mesmos grupos e semente.

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import pandas as pd
from model_research.experiment import paired_group_bootstrap

new = pd.read_csv('results/research/linear-svc-nested/oof.csv')
old = pd.read_csv('results/research/classical-nested/oof.csv')
semantic = pd.read_csv('results/research/semantic-nested/oof.csv')
keys = ['excel_row', 'group_id', 'fold', 'true_label']
assert new[keys].equals(old[keys]) and new[keys].equals(semantic[keys])
pairs = [
    ('selected - clássico anterior', new.pred_selected, old.pred_selected),
    ('selected - semântico completo', new.pred_selected, semantic.pred_selected),
    ('logística nova - anterior', new.pred_word_char_lr, old.pred_selected),
    ('SVM lexical - logística nova', new.pred_word_char_svc, new.pred_word_char_lr),
    ('SVM lexical - SVM palavras', new.pred_word_char_svc, new.pred_word_svc),
]
for name, candidate, reference in pairs:
    result = paired_group_bootstrap(
        new.true_label.to_numpy(), candidate.to_numpy(), reference.to_numpy(),
        new.group_id.to_numpy(), seed=42, repeats=2000,
    )
    print(name, result)
PY
```
