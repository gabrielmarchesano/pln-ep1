# Experimento de limpeza do dataset com o melhor classificador lexical

Protocolo registrado em 23/09/2026, antes de iniciar a execução. O pedido original e os anexos mencionados nas respostas **não estão disponíveis**; esta rodada usa exclusivamente `resp_text` e os rótulos existentes. Nenhuma linha ou rótulo da planilha original será modificado.

## Estado da execução

- Iniciada e concluída em 23/09/2026 no contêiner `pln-cleaning-svc-20260923`, com código de saída 0 e duração registrada de 12,30 minutos.
- Os testes de regressão passaram antes da execução: 48 aprovados, três pulados. A imagem Docker é ROCm, mas este experimento lexical executou na CPU.
- `results.json` indica `status=complete`; a auditoria independente conferiu base, código, 20.092 linhas, rótulos, grupos, folds, seleção interna, filtragem e métricas OOF. Ver [auditoria](../../results/research/dataset-cleaning-svc-20260923/audit.json).

## Objetivo e controle

O maior resultado pontual OOF da rodada clássica anterior foi o LinearSVC com TF-IDF de palavras e caracteres: 45,18% de acurácia agregada (45,19% de média dos folds). O parâmetro `C=0,05` venceu a busca interna nos três folds ([avaliação anterior](avaliacao-linear-svc.md)). A vantagem sobre a regressão logística foi de apenas 0,045 ponto percentual, com intervalo descritivo incluindo zero. Assim, este é o **melhor score observado**, não uma superioridade estabelecida.

O classificador e `C` ficam fixos. O controle é o mesmo modelo treinado no texto original, repetido nas **mesmas novas divisões** usadas pelos tratamentos de limpeza. A comparação histórica com 45,18% será informativa, mas não é a medida primária de ganho, pois o novo agrupamento altera os folds.

## Políticas pré-definidas

| Política | Entrada do modelo | Linhas de treino |
|---|---|---|
| `raw` | Texto original, com a normalização já feita pelo TF-IDF | Todas |
| `masked` | URLs, e-mails e números contíguos de seis ou mais dígitos substituídos por marcadores | Todas |
| `conflicts_excluded` | Texto original | Exclui do **treino do fold** grupos de resposta exata com rótulos divergentes naquele treino |
| `masked_conflicts_excluded` | Texto com marcadores | Mesma exclusão, calculada somente no treino do fold |

O mascaramento preserva negação, acentos, pontuação e números curtos; ele não elimina o restante da resposta. O filtro de conflitos não consulta rótulos de validação, não renomeia classes e não remove exemplos da avaliação. Portanto, trata-se de uma limpeza experimental do conjunto **usado para ajuste**, não de uma nova planilha global com rótulos corrigidos.

## Validação e critério de decisão

- Três folds externos e três internos, estratificados por classe e agrupados pelo texto após o mascaramento. Esse agrupamento conservador impede que respostas que se tornam iguais após limpeza apareçam em treino e validação de qualquer política.
- Em cada treino externo, as quatro políticas são comparadas somente pelos folds internos. A maior acurácia média interna é selecionada; empate favorece `raw`.
- No fold externo, todas as políticas são avaliadas para diagnóstico; a predição `selected` representa a política escolhida internamente. A comparação primária é `selected` contra `raw`, linha a linha, nos mesmos folds. Resultados isolados das outras políticas são exploratórios.
- A avaliação mantém todos os rótulos originais, inclusive grupos conflitantes, e registra acurácia, F1 macro, matriz de confusão, resultados por fold, número de linhas excluídas no treino e diferenças pareadas com intervalo descritivo por grupo.
- Como o modelo, a semente e hipóteses foram influenciados por resultados anteriores, esta rodada **não é confirmação independente**. Um ganho pequeno ou inconsistente não justifica afirmar melhora geral no teste oficial.

Implementação arquivada: [cleaning.py](../../src/model_research/dataset/cleaning.py). Para uma nova execução, use um diretório de saída vazio:

```bash
docker run -d --name pln-cleaning-svc-reproduction \
  -v "$PWD:/workspace" -w /workspace -e PYTHONPATH=/workspace/src \
  pln-rocm:6.4.1 python3 -m dataset_experiments.cleaning \
  --train data/train.xlsx --output-dir results/research/dataset-cleaning-svc-reproduction \
  --outer-folds 3 --inner-folds 3 --seed 42 --threads 2
```

Este comando usa o pacote experimental após a reorganização e um destino novo. A execução histórica usou `clarity.cleaning` e seu código exato permanece no commit `539c741`; o manifest original não foi reescrito.

Os resultados estão em [results.json](../../results/research/dataset-cleaning-svc-20260923/results.json) e [oof.csv](../../results/research/dataset-cleaning-svc-20260923/oof.csv).

## Resultado final

| Política | Acurácia OOF | F1 macro | Acertos | Diferença de acertos contra `raw` |
|---|---:|---:|---:|---:|
| `raw` | **45,26%** | 44,99% | 9.094 | Referência |
| `masked` | 44,95% | 44,69% | 9.031 | −63 |
| `conflicts_excluded` | 45,10% | 44,69% | 9.061 | −33 |
| `masked_conflicts_excluded` | 44,79% | 44,41% | 9.000 | −94 |
| `selected` | 45,16% | 44,89% | 9.073 | −21 |

A validação interna escolheu `raw` nos folds 1 e 2 e `masked` no fold 3. A comparação **primária** `selected − raw` foi de **−0,10 p.p.** (21 acertos a menos), com intervalo descritivo pareado de 95% **[−0,26; +0,05] p.p.**. Portanto, esta rodada **não demonstrou melhoria**. As três políticas de limpeza, consideradas individualmente, também ficaram abaixo do controle no agregado. Filtrar conflitos retirou 1.275, 1.365 e 1.310 linhas dos respectivos treinos externos, sem retirar linhas da avaliação.

O controle `raw` desta rodada (45,26%) não deve ser comparado diretamente aos 45,18% históricos como suposto ganho: o agrupamento por texto mascarado mudou as divisões. A decisão é manter os rótulos e o pré-processamento originais por enquanto; nenhum novo modelo final foi treinado. Os intervalos são descritivos e não incorporam a seleção adaptativa das hipóteses anteriores.
