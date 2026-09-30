# ACH2118 Introdução ao Processamento de Língua Natural

## Modelo para elaboração de relatórios – EP1

> Transcrição detalhada do [PDF do modelo de relatório](ep1-modelo-relatorio.pdf).
> A redação e os exemplos foram preservados conforme o documento original.

O relatório é composto dos itens a seguir, e deve ser entregue em pasta `.ZIP` juntamente com os outros dois itens solicitados (**pdf da apresentação** e **planilha xlxs com os rótulos de teste**).

---

## 1. Nome e número USP

**Nome e número USP** de todos os integrantes efetivos do grupo – esta será a informação levada em conta para eventual aplicação do bônus por ausências/desistências.

---

## 2. Parágrafo introdutório

Um parágrafo introdutório com uma **breve descrição** da estratégia principal escolhida para cada uma das tarefas de classificação, resumindo:

- qual o tipo de representação textual utilizada:
  - palavras;
  - caracteres;
  - *embeddings*;
  - etc.;
- e o tipo de classificador utilizado.

É permitido descrever mais de um modelo para cada tarefa desde que o modelo final escolhido para cada tarefa seja claramente indicado.

---

## 3. Pré-processamento

**Pré-processamento:** como os textos foram tratados antes de treinar o classificador.

---

## 4. Parâmetros avaliados na busca em grade

**Parâmetros** do classificador que foram avaliados na busca em grade: uma tabela com a lista dos parâmetros e a faixa de valores considerada para cada um.

### Exemplo fornecido

| Parâmetro | Valores |
|---|---|
| Tol | `1e-5..-3` |
| Neurônios | `{100,150,200,500}` |

---

## 5. Valores ótimos dos parâmetros

**Valores ótimos** que foram obtidos para cada parâmetro e para cada tarefa de classificação (se houver mais de uma).

### Exemplo fornecido

| Tarefa | Parâmetro | Valor |
|---|---|---:|
| Tarefa1 | Tol | `1e-5` |
| Tarefa1 | Neurônios | `150` |
| Tarefa2 | Tol | `1e-3` |
| Tarefa2 | Neurônios | `300` |

---

## 6. Procedimento

**Procedimento:** se houve separação entre treinamento e teste (e qual a separação usada), e quaisquer outros passos necessários para a reprodução do experimento.

---

## 7. Resultado

**Resultado.** Uma tabela com a métrica solicitada no EP para cada tarefa de classificação e **APENAS PARA O MODELO FINAL DE CADA CLASSE**, ou seja, desconsiderando qualquer outra estratégia alternativa descrita acima.

---

## 8. Link para o repositório de código

**Link para o repositório de código** (`github`, `colab` ou similar).

---

## 9. Instruções para reprodução

**Instruções:** passo-a-passo do que é preciso fazer para reproduzir o resultado a partir do código apresentado:

- bibliotecas;
- dependências;
- etc.

---

## Checklist consolidado do relatório

A partir do modelo fornecido, o relatório deve conter:

1. Identificação de todos os integrantes efetivos do grupo, com nome e número USP.
2. Introdução resumindo a estratégia principal de cada tarefa, a representação textual e o classificador.
3. Descrição do pré-processamento.
4. Tabela dos hiperparâmetros avaliados e respectivas faixas/valores da busca em grade.
5. Tabela dos melhores valores encontrados para cada parâmetro e tarefa.
6. Procedimento experimental suficiente para reprodução, incluindo a separação entre treino e teste, quando aplicável.
7. Resultado final, com a métrica solicitada no EP e somente o modelo final de cada classe/tarefa.
8. Link para o repositório do código.
9. Instruções passo a passo para reproduzir os resultados, incluindo bibliotecas e dependências.

> **Observação:** o documento original escreve “xlxs” no trecho referente à planilha; a grafia foi preservada na transcrição principal.
