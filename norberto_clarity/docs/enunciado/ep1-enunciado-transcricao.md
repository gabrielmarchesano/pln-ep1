# Exercício prático 1

**Disciplina:** ACH2118 Introdução ao Processamento de Língua Natural  
**Autor/professor indicado no material:** Ivandré Paraboni  
**Instituição:** EACH - USP

> Transcrição detalhada do [PDF do enunciado](ep1-enunciado.pdf), organizada por página/slide.
> A redação, inclusive eventuais inconsistências ou erros tipográficos do material original, foi preservada sempre que possível.

---

## Página 1 — Capa

**ACH2118 Introdução ao Processamento de Língua Natural**

# Exercício prático 1

**Ivandré Paraboni**  
**EACH - USP**

---

## Página 2 — Tarefa

# Tarefa

Em sistemas públicos de resposta a requisições do cidadão (e-SIC), deseja-se classificar automaticamente a clareza da resposta que foi fornecida ao usuário.

Para isso, é disponibilizado um córpus de treino (`train.xlsx`) de respostas rotuladas em 3 categorias pelos próprios usuários em uma escala 1..5, em que os escores intermediários são agregados em uma classe única.

### Exemplo visual da planilha de treino

O slide mostra uma pequena planilha com duas colunas:

| Coluna | Cabeçalho | Conteúdo ilustrado |
|---|---|---|
| A | `resp_text` | Textos das respostas |
| B | `clarity` | Rótulos de clareza |

Exemplos de rótulos mostrados na coluna `clarity`:

| Linha ilustrativa | `clarity` |
|---:|---|
| 2 | `c5` |
| 3 | `c1` |
| 4 | `c1` |
| 5 | `c234` |

A tarefa a ser resolvida é uma classificação ternária `{c1,c234,c6}`, rotulando o conjunto de texto de testes (sem rótulos) também fornecido.

> **Observação de transcrição:** o próprio slide apresenta `c5` em uma das linhas da planilha ilustrativa, mas descreve as classes da tarefa como `{c1,c234,c6}`. A inconsistência foi mantida exatamente como aparece no material.

---

## Página 3 — Treino e teste

# Treino e teste

O slide representa visualmente o fluxo do experimento em três etapas:

1. **Cj. de treino**
   - Uma planilha com as colunas `resp_text` e `clarity`.
   - Os textos possuem rótulos de clareza.
   - Na ilustração aparecem exemplos como `c5`, `c1`, `c1` e `c234`.

2. **Modelo treinado**
   - O conjunto de treino é usado para produzir um modelo.
   - O modelo é representado graficamente por uma forma preta, sem indicação de um algoritmo específico.

3. **Rotulação do cj. teste**
   - O modelo treinado é aplicado a uma segunda planilha.
   - A planilha de teste contém a coluna `resp_text`.
   - A coluna `clarity` ainda precisa ser preenchida.
   - Um grande ponto de interrogação (`?`) representa os rótulos desconhecidos a serem previstos.

Representação textual do fluxo:

```text
Cj. de treino
(resp_text + clarity)
        |
        v
Modelo treinado
        |
        v
Rotulação do cj. teste
(resp_text + clarity a prever)
```

---

## Página 4 — O que fazer

# O que fazer

- Use os conjuntos de treino e teste não rotulado para desenvolver um modelo de classificação superior a um **baseline tradicional do tipo regressão logística com contagens TF-IDF**.
  - Com penalidade para desempenho inferior ao *baseline*.

- Use o modelo para obter os rótulos de teste.

---

## Página 5 — O que entregar

# O que entregar

Entrega por um único membro de cada grupo até domingo **04/10** de **3 itens** em uma pasta zipada (`.zip`):

1. Relatório do projeto conforme modelo disponibilizado.
2. Apresentação em formato **pdf** (**não ppt/pptx**) com duração de até **10min**.
3. Planilha do conjunto de teste rotulado.

**Grupos:**  
<https://docs.google.com/spreadsheets/d/146ssklR6dNVHaOkIM4VM1XCbTdND7IS5LlX-x5X_LMw/edit?usp=sharing>

---

## Página 6 — O que será avaliado

# O que será avaliado

**Nota-base:**

- acurácia média de teste: **70%**;
- método inovador / bem elaborado: **20%**;
- relatório: **5%**;
- apresentação em aula: **5%**.

Modelos de **Tf-Idf/BoW com regressão logística, SVM, Naive Bayes e afins** não pontuam no critério **“inovação”**.

**(-) penalidades previstas**

---

## Página 7 — Como será avaliado

# Como será avaliado

- A nota-base é normalizada pelo *ranking* global da turma (eg, o melhor resultado é atribuído nota dez etc.). Este *ranking* determina a ordem de apresentação dos trabalhos.

- Finalmente, são aplicadas as penalidades (problemas de entrega etc.) previstas de forma cumulativa, resultando na nota final do EP.

---

## Página 8 — O que será penalizado

# O que será penalizado

- **Baixo desempenho:** EPs cuja acurácia média seja inferior à média do *baseline* de classe majoritária recebem nota **3,0** e não são avaliados, e também não qualificam para o critério **“inovação”**.

- **Descontos de 20%** aplicado a cada um dos casos a seguir, de forma cumulativa:
  - Resultado inferior ao *baseline* oficial (**regressão logística + TF-IDF**);
  - Não entregar de relatório ou código no prazo;
  - Não entregar do PDF da apresentação;
  - Formato inválido do arquivo de teste (e.g., mais/menos linhas ou colunas), ou rótulos diferentes/ fora da posição, tipo de arquivo não `xlsx` etc.

> No slide, as expressões **“Baixo desempenho”** e **“Descontos de 20%”** aparecem destacadas em vermelho.

---

## Página 9 — O que será valorizado

# O que será valorizado

- Desistências e desaparecimentos podem, opcionalmente, ser indicados no relatório.

- Bônus de **10% na nota final para cada integrante “perdido”**.

- Objetivo é valorizar grupos que tiveram que trabalhar mais.

---

## Página 10 — Cronograma

# Cronograma

- Entrega (**resultado + apresentação + relatório com código**) até domingo **04/10**.

- Apresentação dos trabalhos em aula em **06/10** e **13/10** em ordem de resultado de acurácia do teste – começando pelo último colocado.

- Ausência na hora da apresentação **anula anota da apresentação** (mas não do trabalho como um todo, desde que atenda aos demais critérios).

> **Observação de transcrição:** a expressão “anula anota da apresentação” foi preservada conforme aparece no material.

---

## Página 11 — Avaliação dos resultados

# Avaliação dos resultados

- É possível que haja alguma queda de desempenho entre treinamento e teste.

- Mas uma diferença muito grande indica *overfitting*, e que portanto o modelo não funciona.

- Um resultado espetacular no treinamento, que depois cai drasticamente, não é um resultado real.

- Como se proteger disso?
  - Pensando muito bem em quais características modelar;
  - Realizando um grande número de testes com muitos parâmetros, algoritmos, representações etc. usando muito *grid search*;
  - Realizando validação cruzada.

---

## Página 12 — O que é esperado desta tarefa

# O que é esperado desta tarefa

- Construir sua própria infraestrutura de desenvolvimento, e um **pipeline robusto de experimentação** (pensando no EP-2!).

- Resultados satisfatórios em relação ao *baseline*, e com pouco ou nenhum *overfitting*.

- Despertar para as dificuldades práticas de modelar problemas de PLN usando aprendizado supervisionado.
  - Em um problema simples, com dados balanceados e em pequeno volume.

- Apresentação satisfatória do trabalho realizado na forma escrita e oral.

---

## Página 13 — Dicas finais

# Dicas finais

- **Comece simples:** garanta algum resultado por meio de um método rápido antes de se comprometer com soluções mais sofisticadas.

- Não subestime o **tempo de treino**: mesmo os algoritmos mais simples podem consumir muito tempo de treino, especialmente quando usando *grid search* (altamente recomendado).

- Confira a entrega para se certificar de que **não faltou nada** do que foi pedido.

- Ocupe **todos os integrantes do grupo** para que o trabalho se torne viável.

---

## Página 14 — Encerramento

# Bom trabalho!
