# Experimento prospectivo de hipóteses sobre o dataset

Protocolo definido em 23/09/2026, antes da execução completa. A planilha `data/train.xlsx`, os artefatos de produção e os modelos em `src/model_research/models.py` não serão modificados. O executor independente está em [`src/model_research/dataset/dataset_hypotheses.py`](../../src/model_research/dataset/dataset_hypotheses.py), separado do pacote de produção.

## Escopo e ressalva do holdout

O histórico do projeto já analisou as 20.092 linhas da base. Portanto, um subconjunto reservado **agora** pode ser congelado prospectivamente, mas não se torna um teste historicamente virgem. Ele não será usado para selecionar técnicas, filtrar exemplos, inspecionar conflitos rotulados ou medir score nesta rodada. O arquivo de divisões contém apenas número da linha, grupo, papel e fold de desenvolvimento; rótulos do holdout não são exportados. A avaliação oficial externa, se disponível, continuará sendo necessária.

Primeiro, todos os textos são agrupados por equivalência após normalização ampliada de identificadores e por similaridade cosseno de TF-IDF de caracteres 3–5 com limiar fixo de 0,93. O agrupamento é calculado **sem rótulos**. Com semente 42, uma das cinco partições `StratifiedGroupKFold` é reservada como holdout e as demais formam o desenvolvimento. No desenvolvimento, todas as técnicas usam os mesmos três folds agrupados. O rótulo do holdout entra apenas na estratificação inicial, não em qualquer ajuste ou decisão experimental posterior. O limiar define operacionalmente “quase igual”; a semelhança semântica de paráfrases não é garantida.

## Hipóteses ainda não testadas

O controle é o LinearSVC lexical de palavras/caracteres com `C=0,05` e texto original. A rodada anterior já comparou mascaramento simples de URLs, e-mails e números longos, exclusão integral de conflitos exatos no treino e sua combinação. Esses tratamentos **não são repetidos**.

| Política | Alteração isolada ou combinada |
|---|---|
| `raw` | Controle sem nova intervenção. |
| `majority` | Em cada treino, rótulo majoritário de grupos de texto exatamente igual; empates são removidos apenas desse treino. |
| `normalized` | Substituição de URL, e-mail, CPF, CNPJ, data, valor monetário, protocolo/processo e números por marcadores. |
| `style` | Acrescenta número de palavras, frases, parágrafos, média de palavras por frase, diversidade lexical, comprimento médio, pontuação e proporções de dígitos/maiúsculas. Escala ajustada só no treino. |
| `ordinal` | Duas regressões logísticas para limiares `y ≥ c234` e `y ≥ c5`; probabilidades são projetadas para preservar a ordem e convertidas em três classes. |
| `consensus_filter` | Remove do treino exemplos cuja anotação diverge do consenso OOF de SVC, regressão logística e Naive Bayes, exigindo probabilidade ≥ 0,65 na logística. |
| `combined_svc` | Normalização, estilo, maioria e filtro de consenso juntos, com LinearSVC. |
| `combined_ordinal` | Mesma combinação, com classificador ordinal. |

O detector de consenso gera suas predições OOF **dentro de cada treino externo**, com folds agrupados próprios. Ele jamais usa rótulos do fold de validação. Divergência não é prova de erro de anotação; as linhas sinalizadas são excluídas somente do ajuste experimental. Os conflitos de grupos semelhantes são diagnosticados somente na partição de desenvolvimento. O controle e todos os tratamentos são medidos em exatamente as mesmas linhas de validação, com acurácia e F1 macro agregados e por fold. A avaliação do holdout fica deliberadamente pendente.

## Execução e arquivos

```bash
docker run --rm --name pln-dataset-hypotheses-reproduction \
  -v "$PWD:/workspace" -w /workspace -e PYTHONPATH=/workspace/src \
  pln-rocm:6.4.1 python3 -m dataset_experiments.dataset_hypotheses \
  --train data/train.xlsx --output-dir results/research/dataset-hypotheses-reproduction \
  --folds 3 --holdout-folds 5 --seed 42 --similarity-threshold 0.93 --threads 2
```

Este comando usa o pacote experimental após a reorganização e um destino novo. A execução histórica usou `clarity.dataset_hypotheses` e seu código exato permanece no commit `539c741`; o manifest original não foi reescrito.

O executor recusa diretórios de saída não vazios, grava `manifest.json` e `splits.csv` antes de treinar, depois salva diagnósticos por fold, `development-oof.csv` e `results.json`. O hash SHA-256 da planilha é conferido antes e após a execução. Se houver interrupção, os arquivos parciais permanecem para auditoria e a execução não os sobrescreve automaticamente.

## Resultado

Execução concluída em 23/09/2026, com `status=complete` em 16,07 minutos. A [auditoria independente](../../results/research/dataset-hypotheses-20260923/audit.json) passou: hash da planilha intacto (`0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`), divisões e rótulos conferidos, 16.093 predições OOF de desenvolvimento reproduzidas e nenhuma sobreposição de grupos. As 3.999 linhas do holdout ficaram sem predição ou score; o conjunto é prospectivo, não historicamente virgem.

Os agrupamentos textuais formaram 16.237 componentes na base inteira. No desenvolvimento, 440 componentes continham rótulos conflitantes em 3.170 linhas; 364 desses componentes reuniam mais de um texto exato. O limiar cosseno de 0,93 e a canonicalização de identificadores são regras operacionais: podem unir textos distintos ou deixar paráfrases separadas. Os números descrevem candidatos a revisão, não erros de anotação confirmados.

| Política | Acurácia OOF | F1 macro OOF | Diferença de acurácia vs. `raw` | Acertos líquidos |
|---|---:|---:|---:|---:|
| `raw` | **44,35%** | 43,73% | Referência | 7.137 acertos |
| `majority` | 44,19% | 43,57% | −0,16 p.p. | −26 |
| `normalized` | 44,21% | 43,65% | −0,14 p.p. | −23 |
| `style` | 44,39% | 43,80% | +0,04 p.p. | +7 |
| `ordinal` | 43,60% | 42,30% | −0,75 p.p. | −120 |
| `consensus_filter` | 44,34% | 43,70% | −0,01 p.p. | −1 |
| `combined_svc` | **44,50%** | **43,85%** | +0,16 p.p. | +25 |
| `combined_ordinal` | 43,57% | 42,53% | −0,78 p.p. | −125 |

A combinação SVC ganhou 0,43 e 0,67 p.p. nos folds 1 e 2, mas perdeu 0,65 p.p. no fold 3. Contra `raw`, corrigiu 743 predições antes erradas e passou a errar 718 antes corretas. Seu intervalo descritivo pareado de 95% por componente textual para o delta de acurácia foi **[−0,34; +0,62] p.p.**; inclui zero e não corrige a comparação exploratória de oito políticas. O recall de `c234` caiu de 32,21% (`raw`) para 31,07% (`combined_svc`), apesar do pequeno ganho agregado. Os atributos de estilo isolados ganharam apenas sete acertos, com intervalo **[−0,29; +0,37] p.p.**

A classificação ordinal caiu em todos os folds; no agregado, perdeu 120 acertos e seu F1 macro caiu 1,43 p.p. frente ao controle. A política de maioria removeu 170/248/246 linhas empatadas dos três treinos e reatribuiu 342/416/412 rótulos, sem melhora. O consenso OOF sinalizou apenas 25/59/39 linhas para remoção nos três treinos externos e não alterou o resultado materialmente. Divergência de modelos não identifica automaticamente um rótulo incorreto.

**Decisão final:** continuar com a planilha original `data/train.xlsx`, sem alterar textos ou rótulos e sem incorporar qualquer tratamento experimental ao pipeline de produção. A combinação SVC é a maior pontuação pontual desta nova divisão, mas o efeito é pequeno, instável e sem evidência pareada robusta. O controle de 44,35% não é diretamente comparável aos 45,26% da limpeza anterior nem aos 45,18% históricos, porque holdout e agrupamentos alteraram os folds. O teste congelado permanece sem avaliação; qualquer uso futuro requer uma regra final pré-fixada e não desfaz o fato de que a base completa já foi explorada antes desta reserva.

Artefatos: [resultado completo](../../results/research/dataset-hypotheses-20260923/results.json), [divisões congeladas](../../results/research/dataset-hypotheses-20260923/splits.csv), [predições OOF de desenvolvimento](../../results/research/dataset-hypotheses-20260923/development-oof.csv) e [auditoria](../../results/research/dataset-hypotheses-20260923/audit.json).
