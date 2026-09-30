# Diagnóstico do dataset de clareza e oportunidades de melhoria

**Diagnóstico e plano históricos.** As hipóteses de tratamento foram testadas sem ganho estável; o dataset original foi mantido. O modelo escolhido posteriormente é o [NorBERTo-512 LoRA](../modelo-escolhido-norberto-512.md). Consulte o [índice](README.md) para distinguir análise, experimentos e avaliação final.

Data da análise: 23/09/2026. Base: `data/train.xlsx` (SHA-256 `0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`). Este relatório usa somente leitura da planilha e das predições fora do treino (OOF) já produzidas; não altera rótulos nem apresenta uma nova avaliação de modelo.

**Restrição operacional atual:** não teremos acesso ao pedido original nem aos anexos. As referências a esse contexto abaixo descrevem uma limitação do dataset, não uma etapa de coleta ou um recurso do modelo. O plano vigente usa apenas os arquivos já disponíveis.

## Resumo executivo

O gargalo não parece ser apenas a escolha do encoder. A regressão logística com TF-IDF de palavras e caracteres atingiu **45,14% de acurácia OOF**; o NorBERTo LoRA, com contexto 256 e três folds externos, atingiu **41,52%** no mesmo particionamento. Há rótulos diferentes para textos idênticos, respostas padronizadas que remetem a anexos ausentes da planilha e uma classe intermediária de difícil separação. Melhorar a informação disponível e a consistência dos rótulos merece prioridade sobre aumentar o modelo ou a janela de tokens.

Os rótulos foram originalmente atribuídos pelos próprios usuários em escala de 1 a 5; `c234` agrega as notas intermediárias 2, 3 e 4 ([enunciado](../enunciado/ep1-enunciado-transcricao.md)). A planilha de treino contém **somente** `resp_text` e `clarity`: não há pergunta original, conteúdo de anexo, identificação do avaliador ou nota original dentro de `c234`. Esses dados não podem ser inferidos de forma confiável apenas da resposta.

## Evidências do dataset

| Medida | Resultado | Interpretação |
|---|---:|---|
| Respostas | 20.092 | Volume limitado para ajuste supervisionado de encoder. |
| Classes | `c1`: 6.347; `c234`: 6.853; `c5`: 6.892 | O desbalanceamento global é pequeno. |
| Textos normalizados únicos | 18.143 | Entidades HTML, espaços e caixa são normalizados; acentos e pontuação permanecem. |
| Grupos de texto exato repetido | 500 grupos / 2.449 linhas | A mesma resposta visível aparece várias vezes. |
| Grupos com rótulos conflitantes | 316 grupos / 1.975 linhas | 9,83% das linhas pertencem a uma resposta repetida com mais de um rótulo. |
| Divergências mínimas para uma função determinística do texto normalizado | 919 linhas / 4,57% | Soma dos rótulos não majoritários em cada grupo. O **limite empírico de concordância de 95,43% nesta amostra** não é um teto de generalização nem uma acurácia OOF alcançável. |
| Comprimento | mediana 101; percentil 95 de 419; máximo de 1.821 palavras | Há uma cauda longa de respostas. |
| Menção a anexo, arquivo ou documento | 6.783 linhas / 33,76% | Conteúdo relevante pode estar fora de `resp_text`. |
| URL na resposta | 7.936 linhas / 39,50% | Links podem apontar para informações indisponíveis ao modelo. |

O maior grupo de respostas exatamente iguais contém **138 linhas**, distribuídas em `c1=60`, `c234=50` e `c5=28`; sua resposta remete a informação anexa. Isso mostra que o texto visível não determina o rótulo observado. A planilha, sozinha, não permite distinguir se a divergência decorre de anexos diferentes, pedidos diferentes, expectativas dos usuários ou inconsistência de anotação.

Entre as **6.783** linhas que mencionam anexos, arquivos ou documentos, **993 (14,64%)** pertencem a grupos de rótulos conflitantes. Nas outras **13.309** linhas, são **982 (7,38%)**. Essa associação é compatível com contexto ausente, mas não estabelece causalidade: respostas padronizadas e outros fatores podem explicar parte da diferença. Substituir automaticamente os rótulos pelo majoritário de cada grupo seria injustificado.

### Semelhança além de cópias exatas

Os folds atuais isolam respostas iguais após a normalização descrita acima. Um diagnóstico exploratório adicional substituiu URLs e números por marcadores e removeu pontuação. Isso reduziu os grupos distintos de **18.143 para 17.759**. Sob essa canonicalização agressiva, **165 grupos, somando 896 linhas**, aparecem em mais de um fold externo. São candidatos a inspeção manual, **não vazamento comprovado**: pontuação, números e links podem mudar o significado, e a regra pode unir respostas sem relação. Qualquer agrupamento mais forte deve ser definido após inspeção baseada somente no texto, antes de avaliar rótulos ou modelos sob essa regra.

## Onde os modelos falham

A tabela usa predições OOF alinhadas às 20.092 linhas do Excel. Grupos de texto normalizado idêntico permanecem inteiros em cada fold externo. `Lexical` é regressão logística com TF-IDF de palavras/caracteres e `C=0,25`; `NorBERTo` é o experimento completo de contexto 256. Ver [resultados lexicais](../../results/research/linear-svc-nested/results.json) e [resultados NorBERTo](../../results/norberto/experiments/norberto-lora-fulltrain-experiment-lr1e3-ctx256/results.json).

| Conjunto | Linhas | Acurácia lexical | Acurácia NorBERTo |
|---|---:|---:|---:|
| Respostas únicas | 17.643 | 45,17% | 41,85% |
| Repetidas, rótulo consistente | 474 | 56,96% | 47,68% |
| Repetidas, rótulos conflitantes | 1.975 | 42,03% | 37,11% |
| Todas | 20.092 | 45,14% | 41,52% |

O modelo lexical comete **11.023** erros OOF. Apenas **919** erros seriam inevitáveis para um modelo hipotético que memorizasse o rótulo majoritário de cada resposta exata *nessa mesma amostra*. Portanto, conflitos exatos importam, mas não explicam, sozinhos, a maior parte dos erros. Lexical e NorBERTo erram simultaneamente **8.090** linhas; esses casos merecem revisão qualitativa, não a presunção de que todos os rótulos estejam errados.

A classe intermediária é o ponto mais fraco do lexical: recall de **36,45%** (`2.498/6.853`). Das 4.355 respostas `c234` restantes, **1.881** foram previstas como `c1` e **2.474** como `c5`. Os recalls de `c1` e `c5` são **44,02%** e **54,80%**. Agregar três notas subjetivas em `c234` possivelmente cria um alvo heterogêneo, mas a matriz de confusão não comprova, sozinha, esse mecanismo de rotulagem.

O comprimento está associado ao rótulo: `c5` corresponde a **41,6%** das respostas com menos de 32 palavras, mas a apenas **23,5%** daquelas com pelo menos 512 palavras. Ainda assim, as probabilidades OOF do lexical já acompanham razoavelmente essas frequências por faixa. Sua acurácia é **41,74%** nas 575 respostas com pelo menos 512 palavras, contra **45,14%** no total. Incluir comprimento explícito é, portanto, uma ablação barata, não uma solução presumida. O NorBERTo trunca **35,74%** das linhas no limite de 256 tokens; o piloto interno pareado com contexto 512 caiu de **44,71% para 44,52%** e quase dobrou o tempo ([histórico experimental](../desenvolvimento.md)). As evidências atuais não favorecem priorizar contexto maior.

## Revisão da implementação

1. **Validação lexical:** [data.py](../../src/clarity/data.py) cria grupos por hash do texto normalizado; [experiment.py](../../src/model_research/experiment.py) usa folds externos estratificados por grupo e busca interna. O TF-IDF é ajustado somente em cada divisão de treino. Não foi encontrado vazamento de cópias exatas. Respostas semelhantes, mas não idênticas, ainda podem cruzar folds.
2. **Comparação dos encoders:** [finetune.py](../../src/clarity/finetune.py) usa um único holdout interno por fold externo para selecionar checkpoint e treina o encoder em aproximadamente 80% das linhas do treino externo; não faz reajuste em todo o treino externo. O controle lexical treinado nas **mesmas linhas** alcançou **44,70%**, ainda acima dos **41,52%** do NorBERTo. Menor volume de treino, portanto, não é a única explicação, embora a comparação com o controle lexical completo não seja pareada por volume de treino.
3. **Estabilidade da otimização:** as acurácias externas do NorBERTo foram **44,43%**, **36,39%** e **43,90%**. No segundo fold, o checkpoint escolhido foi o da primeira época, e foram registrados gradientes pré-clipping acima de 1.000. Testar taxa de aprendizado menor e estabilidade em piloto interno é razoável antes de outra rodada cara. Essa hipótese de treinamento é distinta das limitações do dataset.
4. **Seleção adaptativa:** as configurações de encoder foram escolhidas após pilotos e observação de resultados anteriores; a rodada completa de três folds é exploratória, não um teste final independente. Qualquer ganho de limpeza, atributos ou arquitetura deve ser selecionado na validação interna e confirmado em dados intocados. Usar todos os rótulos para corrigir a base **antes** da validação cruzada vazaria informação da validação.

## Plano de investigação, em ordem de prioridade

1. **Auditar a integridade do texto e dos rótulos disponíveis.** Examinar amostra estratificada de conflitos exatos, respostas muito curtas, artefatos de codificação, identificadores, modelos de resposta e erros de alta confiança. Registrar as conclusões sem divulgar dados pessoais. Como pedido e anexo não estão disponíveis, um conflito entre rótulos não autoriza escolher um novo rótulo “correto”; marcar casos ambíguos como inconclusivos.
2. **Verificar agrupamentos por similaridade usando apenas o texto.** Inspecionar uma amostra dos 165 grupos canonicalizados que cruzam folds, incluindo falsas uniões causadas por números ou URLs distintos. Fixar qualquer regra conservadora antes de medir seu efeito e manter o mesmo agrupamento para controle e tratamento.
3. **Resultado da [rodada de limpeza concluída](experimento-limpeza-dataset.md).** Com LinearSVC lexical `C=0,05`, as políticas de mascaramento, exclusão de conflitos somente no treino e combinação não superaram o texto original nos mesmos novos folds. A seleção interna obteve 45,16% contra 45,26% de `raw` (−21 acertos; intervalo descritivo incluindo zero). Todos os rótulos originais foram mantidos na avaliação. O agrupamento mudou em relação ao histórico, portanto não interpretar a diferença entre 45,26% novos e 45,18% antigos como ganho.
4. **Se a limpeza não melhorar, testar somente hipóteses textuais específicas.** Exemplos são remover saudações/rodapés repetitivos com regra pré-definida, reduzir o peso de respostas muito repetidas ou acrescentar poucos atributos de forma. Cada hipótese exige ablação contra o melhor controle nos mesmos folds e transformação disponível também na predição. Não ampliar automaticamente a busca de modelos ou corrigir rótulos por maioria global.
5. **Escolher o procedimento final com base na auditoria concluída.** Não promover a limpeza sem evidência de ganho; manter o texto e os rótulos originais até uma hipótese nova e pré-definida demonstrar vantagem pareada. A escolha de um pipeline final, seu ajuste em toda a base e a avaliação no teste oficial ainda estão pendentes.

Intervalos calculados a partir das predições OOF existentes são descritivos e não incorporam toda a incerteza da seleção adaptativa. A pergunta central é o que pode ser previsto a partir do texto efetivamente fornecido, sem pressupor conteúdo de anexos ausentes.

## Conclusão após os experimentos de tratamento

As propostas deste relatório foram testadas nas [rodadas de limpeza](experimento-limpeza-dataset.md) e de [hipóteses do dataset](experimento-hipoteses-dataset.md), sempre contra um controle lexical nos mesmos folds de cada rodada. Mascaramento simples, exclusão de conflitos, voto majoritário, normalização ampliada, estilo, filtro por consenso e classificação ordinal não produziram melhoria estável. A melhor combinação teve apenas 25 acertos líquidos adicionais em 16.093 linhas de desenvolvimento, perdeu no terceiro fold e apresentou intervalo descritivo pareado que inclui zero.

**Decisão registrada:** seguir com `data/train.xlsx` e seus textos e rótulos originais. Não corrigir duplicatas automaticamente, não remover linhas sinalizadas pelos modelos e não aplicar normalização experimental no pipeline de produção. Os executores dessas investigações foram isolados em [model_research.dataset](../../src/model_research/dataset/README.md), enquanto relatórios, predições e auditorias permanecem arquivados. O holdout reservado de 3.999 linhas não foi avaliado durante essas investigações; ele foi consultado depois e passou a ser validação fixa reutilizada. Como a base inteira já havia sido estudada antes da reserva, ele não constitui confirmação historicamente independente.

Os diagnósticos de conflitos, contexto ausente e erros da classe intermediária continuam válidos como limitações observadas, mas não autorizam alterar os rótulos sem evidência externa. Uma eventual avaliação oficial rotulada continua sendo a principal evidência futura de generalização.
