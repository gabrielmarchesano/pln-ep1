# Classificação de clareza

O modelo escolhido para a entrega é o **NorBERTo-base com LoRA e contexto de 512 tokens**. O código de uso fica em `src/clarity/`; os experimentos com outros modelos e tratamentos do dataset ficam em `src/model_research/`, `configs/research/`, `tests/research/` e `docs/research/`. Os resultados estão separados em [`results/norberto/`](results/norberto/README.md) e [`results/research/`](results/research/README.md), com os manifests e resultados históricos preservados.

A planilha pronta é [deliveries/test1.xlsx](deliveries/test1.xlsx), com 900 valores de `clarity` preenchidos. O arquivo original `data/test1.xlsx` não foi modificado. Consulte a [auditoria da entrega](docs/norberto/entrega-test1-norberto-512.md) e a [decisão do modelo](docs/modelo-escolhido-norberto-512.md).

## Instalação e execução direta

Use Python 3.11 e instale as dependências em um ambiente virtual:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` é o ambiente do NorBERTo escolhido. As bibliotecas compartilhadas estão em `requirements/common.txt`; testes em `requirements/dev.txt`; e dependências opcionais de modelos alternativos em `requirements/research/`. Use ambientes virtuais separados para novas pesquisas, evitando misturar pilhas de modelos.

Instale uma distribuição de PyTorch adequada ao ambiente (CPU, NVIDIA CUDA ou AMD ROCm). **Recomendamos fortemente executar tanto o treino quanto a inferência em GPU NVIDIA/CUDA ou AMD/ROCm para acelerar o processamento**, especialmente o treino completo; CPU é uma alternativa funcional, mas potencialmente muito mais lenta. `train` e `test` aceitam `--device auto|cpu|gpu`: `auto` prefere GPU e usa CPU quando indisponível; `cpu` força CPU; `gpu` exige uma GPU funcional e identifica CUDA ou ROCm internamente, sem fallback silencioso. A saída informa o dispositivo selecionado, o backend e o nome da GPU, quando houver. Não é necessário Docker.

**Adapter vigente de entrega:** [`artifacts/norberto-lora-final-ctx512-20260928/final_model/`](artifacts/norberto-lora-final-ctx512-20260928/final_model/). O adapter, o tokenizer e os metadados de verificação estão versionados: após clonar o repositório, **não é necessário treinar novamente** para usar `test`. O adapter posterior do `candidate-08` **não** foi promovido, pois a pequena melhora exploratória na CV não se confirmou na avaliação do modelo salvo. Os pesos do **NorBERTo-base** não estão no Git; a revisão fixada é baixada automaticamente na primeira inferência, portanto esse primeiro uso requer acesso ao Hugging Face ou um cache já preenchido.

Para preencher uma nova planilha com coluna `resp_text` e `clarity` vazia:

```bash
.venv/bin/python src/main.py test \
  --test data/test1.xlsx \
  --output deliveries/test1-nova-execucao.xlsx \
  --output-dir results/norberto/delivery/prediction-test1-nova-execucao
```

Se `--output` for omitido, o destino padrão é `deliveries/<nome do arquivo de entrada>`; uma saída já existente nunca é sobrescrita. `test` só executa inferência, preserva as demais células e produz CSV de predições, probabilidades e metadata no diretório de resultados indicado. A planilha sem gabarito não permite calcular acurácia.

Para treinar um **novo** adapter a partir de `data/train.xlsx`, excluindo o holdout congelado e reutilizando as épocas selecionadas pelos três folds já concluídos:

```bash
.venv/bin/python src/main.py train \
  --train data/train.xlsx \
  --output-dir results/norberto/delivery/norberto-final-nova-execucao \
  --artifact-dir artifacts/norberto-final-nova-execucao
```

O treinamento não avalia o holdout, não reutiliza o adapter de nenhum fold e exige diretórios de saída novos. A execução em GPU CUDA/ROCm é fortemente recomendada; em CPU pode exceder o limite operacional de 75 minutos da configuração aprovada e terminar incompleta, por isso uma execução integral em CPU não foi validada. Para usar o novo adapter em `test`, passe `--artifact-dir artifacts/norberto-final-nova-execucao`. Os comandos esperam ser executados a partir da raiz do projeto.

## Evidências e histórico

O NorBERTo-512 obteve 43,47% de acurácia nos três folds de desenvolvimento e 45,31% no holdout fixo de 3.999 linhas. A regressão logística lexical obteve 45,09% nesse holdout; a diferença não demonstra superioridade estatística. A escolha também considerou o critério de inovação. O holdout havia sido consultado em pesquisas anteriores, portanto não é um teste historicamente intocado.

O [índice de documentação](docs/README.md) distingue decisão, avaliação, treino final, entrega e pesquisa. O [índice de resultados](results/README.md) identifica o adapter vigente e separa os testes não promovidos. O [desenvolvimento](docs/desenvolvimento.md) mantém a cronologia; os [relatórios de outros modelos](docs/research/README.md) e o [código experimental](src/model_research/) continuam disponíveis para novos testes, sem serem a implementação principal.
