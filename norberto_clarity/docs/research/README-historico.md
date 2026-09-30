# EP1 — Classificação de clareza de respostas do e-SIC

Pipeline para classificar respostas em `c1`, `c234` e `c5`, com comparação ao baseline, seleção de parâmetros, avaliação sem vazamento de duplicatas e exportação de Excel.

O enunciado menciona `c6`, mas a planilha contém `c5`. Usamos os rótulos reais, sem renomear classes. Os requisitos e o modelo de relatório estão no [guia da documentação](../README.md).

**Modelo escolhido:** [NorBERTo-base com LoRA e contexto de 512 tokens](../modelo-escolhido-norberto-512.md), treinado novamente nas 16.093 linhas de desenvolvimento do dataset original. O adapter e o tokenizer estão salvos localmente em `artifacts/norberto-lora-final-ctx512-20260928/final_model/`. Nos três folds de desenvolvimento, obteve **43,47% de acurácia**; na avaliação separada do holdout fixo de 3.999 linhas, **45,31% de acurácia e 45,31% de F1 macro**. A regressão logística lexical obteve 45,09% de acurácia nesse mesmo holdout. A diferença de nove acertos não demonstra superioridade geral; o NorBERTo foi escolhido também pelo critério de inovação. O holdout já foi consultado em pesquisas anteriores e não é um teste final independente.

**Planilha pronta:** [baixar `test1-norberto-512.xlsx`](../../artifacts/test1-norberto-512.xlsx), com as 900 células de `clarity` preenchidas pelo modelo escolhido. Veja [comando, hashes e auditoria](../norberto/entrega-test1-norberto-512.md). **Leia também:** [decisão e artefato](../modelo-escolhido-norberto-512.md) · [registro de desenvolvimento](../desenvolvimento.md) · [avaliação dos três folds e treino final](../norberto/avaliacao-norberto-512-e-treino-final.md) · [avaliação no holdout](../norberto/avaliacao-norberto-holdout-final.md). O [índice completo](../README.md) separa avaliações, experimentos e histórico.

## Instalação

Use Python 3.11–3.13. As rodadas clássicas foram executadas em Python 3.13 e a inferência NorBERTo-512 foi validada em Python 3.12. Para o **modelo escolhido**, instale as dependências de ajuste/inferência em um ambiente virtual; para GPU, use um wheel PyTorch 2.8.0 compatível com seu driver e backend antes de executar o comando.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No Windows, ative com `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`). Execute os comandos a partir da raiz do projeto. Para testes, instale também `requirements/dev.txt`; para usar apenas os modelos clássicos, basta `requirements/common.txt`. O adapter final escolhido e `data/test1.xlsx` agora estão versionados no Git; os pesos-base precisam ser baixados ou estar em cache.

## Auditar os dados

```bash
PYTHONPATH=src python -m model_research.cli inspect --train data/train.xlsx
```

A auditoria registra classes, textos vazios, repetições, conflitos de rótulos, comprimentos e SHA-256 do arquivo. O treino tem 20.092 respostas e 316 grupos de respostas repetidas com rótulos conflitantes. As linhas são preservadas. A célula numérica da linha 5028 é convertida em texto para o modelo e registrada no diagnóstico. A análise de duplicatas, rótulos e informação ausente está no [relatório do dataset](analise-problemas-dataset.md).

## Comparar e selecionar modelos

```bash
PYTHONPATH=src python -m model_research.cli experiment \
  --train data/train.xlsx \
  --config configs/research/classical.json \
  --folds 3 --inner-folds 3 --seed 42 --threads 2 \
  --output-dir results/research/meu-experimento
```

O experimento usa validação cruzada **aninhada**:

1. Os folds externos medem o desempenho em exemplos não usados para selecionar parâmetros.
2. Dentro de cada treino externo, a busca em grade usa outros folds para selecionar os parâmetros e o modelo por acurácia.
3. O vencedor interno é treinado no treino externo completo e avaliado no fold externo.
4. O procedimento repete até cada linha ter uma predição fora do treino, chamada OOF.

Nos dois níveis, respostas iguais após normalização pertencem ao mesmo grupo. TF-IDF aprende vocabulário e IDF somente no treino de cada divisão. Não removemos exemplos conflitantes nem usamos rótulos de validação para corrigir duplicatas. Textos apenas parecidos ainda podem cruzar folds.

O relatório JSON contém scores internos, parâmetros, métricas externas por classe, matriz de confusão, acurácia de treino, diferença treino/validação, versões e configuração. Novas execuções também registram tempos de ajuste/scoring por configuração, tempo do reajuste vencedor e seus diagnósticos de iterações/perda, quando disponíveis. `oof.csv` contém linha original do Excel, grupo, fold, rótulo real e predições. `fold-N.json` preserva resultados parciais. Use um novo diretório por execução; resultados existentes não são sobrescritos.

A série **selected** avalia o procedimento completo de seleção. Escolher uma família olhando seus resultados externos e reportar apenas o melhor score introduz viés. Os intervalos por reamostragem de grupos são descritivos, não uma garantia de superioridade no teste oficial.

Para reproduzir os modelos fixos da versão anterior:

```bash
PYTHONPATH=src python -m model_research.cli evaluate --train data/train.xlsx --model baseline --folds 3 --output results/research/baseline-fixo.json
PYTHONPATH=src python -m model_research.cli evaluate --train data/train.xlsx --model candidate --folds 3 --output results/research/svm-fixo.json
```

O agrupamento agora usa hashes dos textos normalizados. Isso mantém as mesmas relações de duplicidade, mas altera a ordenação dos grupos e, portanto, as partições frente à versão inicial. Compare modelos nos mesmos folds; não compare diretamente com métricas obtidas nas partições anteriores.

## Resultados históricos dos modelos clássicos e semânticos congelados

As seis grades, incluindo a rodada MLP concluída em 17/09/2026, estão completas: 3 folds externos × 3 internos, semente 42, 20.092 predições OOF por experimento e nenhuma sobreposição de grupos entre folds externos.

| Experimento | Acurácia média de `selected` |
|---|---:|
| [MiniLM + MLP e controles](../../results/research/semantic-mlp-nested/results.json) — selecionou logística lexical | 45,14% |
| [LinearSVC + logística regularizada](../../results/research/linear-svc-nested/results.json) | 45,02% |
| [Clássico](../../results/research/classical-nested/results.json) | 44,69% |
| [Semântico completo](../../results/research/semantic-nested/results.json) | 44,62% |
| [Semântico compacto](../../results/research/semantic-compact/results.json) | 44,02% |
| [Estático/POTION](../../results/research/static-nested/results.json) | 43,83% |
| Baseline fixo, comum às seis execuções | 43,87% |

`selected` é o modelo escolhido internamente em cada fold, incluindo a possibilidade de escolher o baseline. Na nova rodada, SVM palavras/caracteres obteve 45,19% como família e a logística mais regularizada, 45,14%; a diferença agregada foi de apenas 9 acertos em 20.092 linhas. Não há evidência clara de superioridade do SVM sobre essa logística. O ganho do novo `selected` frente ao clássico anterior também é incerto. Os embeddings isolados ficaram abaixo do baseline. Veja o [histórico](../desenvolvimento.md) e a [análise LinearSVC com plano de experimentos e custos](avaliacao-linear-svc.md).

A MLP isoladamente obteve **43,68%**, contra 42,58% do MiniLM + logística e 45,14% da logística lexical. `selected` escolheu a logística lexical nos três folds; não atribua seus 45,14% à MLP. A análise de convergência, custos e intervalos está no [relatório MLP](experimento-semantic-mlp.md).

A [referência lexical](modelo-final-logistico.md) está salva localmente em `artifacts/final-logistic-20260928.joblib`: regressão logística com `C=0,25`, reajustada nas 20.092 linhas e auditada. Ela **não é o modelo escolhido**. `artifacts/classical.joblib`, com `C=0,5`, permanece apenas como artefato anterior.

## Rodada concluída: LinearSVC

A execução original levou 30,69 minutos. O plano recomendado reduz a grade antes de repetir sementes; não é necessário repetir esta busca ampla. Para reprodução deliberada, use um diretório novo:

```bash
PYTHONPATH=src python -m model_research.cli experiment \
  --train data/train.xlsx --config configs/research/linear-svc.json \
  --folds 3 --inner-folds 3 --seed 42 --threads 2 --verbose 2 \
  --output-dir results/research/linear-svc-reproducao
```

Comparamos LinearSVC só com palavras e com palavras/caracteres, usando exatamente os mesmos vetorizadores das referências logísticas. Para cada SVM, a grade de C é `{0.01, 0.05, 0.1, 0.25, 0.5, 1.0}`; C menor significa regularização mais forte. O solver usa `dual="auto"`, tolerância `1e-4`, limite de 10.000 iterações e semente fixa. Esta rodada usa CPU e não precisa calcular novos embeddings.

A grade inclui o baseline fixo e a regressão logística de palavras/caracteres com C em `{0.1, 0.25, 0.5}`. São 16 configurações, 144 avaliações internas, mais os reajustes dos vencedores. O candidato SVM antigo permanece disponível como `candidate`; os novos modelos mantêm os vocabulários maiores da representação lexical atual.

`--verbose 2` imprime cada ajuste. `manifest.json` registra a configuração, o hash da base e o código no início; `fold-N.json` registra folds concluídos; `results.json` e `oof.csv` indicam a conclusão do experimento. Para repetir, escolha outro diretório de saída.

## Rodada concluída: MiniLM congelado + MLP pequena

A rodada testou trocar o classificador linear por uma rede neural pequena, mantendo exatamente os mesmos embeddings congelados. O encoder não é treinado novamente. A arquitetura é `384 → 64 (ReLU) → 3`, com Adam, até 200 épocas e dois valores de regularização L2 (`alpha=0.1` e `1.0`). O treinamento usa CPU; esta implementação do scikit-learn não usa CUDA. A execução levou 22,89 min; para reprodução deliberada:

```bash
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src python -u -m model_research.cli experiment \
  --train data/train.xlsx --config configs/research/semantic-mlp.json \
  --folds 3 --inner-folds 3 --seed 42 --threads 2 --verbose 2 \
  --output-dir results/research/semantic-mlp-reproducao
```

O comando pressupõe o encoder já baixado; neste ambiente o cache de embeddings está completo. Para repetir, use outro diretório. Modo offline evita downloads, mas não proíbe calcular embeddings faltantes com pesos locais.

A grade inclui baseline, logística lexical fixa C=0,25 e MiniLM + logística com C em `{1, 10, 100}`. São sete configurações, **63 ajustes internos + 12 reajustes das famílias**, além das três referências majoritárias. Só a MLP tem duas configurações; os controles não podem ser omitidos da comparação.

`early_stopping=False` evita a divisão aleatória de validação interna do scikit-learn, que não respeita grupos. A parada usa somente a perda de treino (`tol=1e-4`, paciência de 15 épocas) ou o limite de épocas. Os folds internos escolhem `alpha`; os externos não controlam treinamento nem parada. Avisos de limite de iterações devem ser examinados no log, não interpretados como convergência assegurada.

Veja os [resultados, protocolo e piloto da rodada MLP](experimento-semantic-mlp.md). Alpha=0,1 venceu nos três folds, mas houve dez avisos de limite de iterações e a MLP não superou a referência lexical. Depois foram concluídos os pilotos BERTimbau e NorBERTo, inclusive a [avaliação completa do NorBERTo-512 no desenvolvimento](../norberto/avaliacao-norberto-512-e-treino-final.md). A limpeza do dataset não superou o controle lexical no mesmo protocolo. A logística lexical de 45,14% permanece uma referência histórica forte; os protocolos e partições de épocas diferentes não são diretamente comparáveis.

## Modelos disponíveis

O ajuste supervisionado do BERTimbau usa um runner separado e um protocolo de holdout interno, descritos em [experimento-bertimbau.md](experimento-bertimbau.md). Ele não reutiliza o cache de embeddings nem deve ser confundido com a validação aninhada das famílias abaixo.

O primeiro piloto supervisionado obteve **38,33% de acurácia interna**. A segunda rodada, alterando somente a taxa de aprendizado de 0,0002 para 0,001, chegou a **39,82%**, contra **41,50% do lexical treinado nas mesmas linhas**. O ganho entre pilotos foi de 1,49 p.p., mas seu intervalo descritivo pareado inclui zero e o recall de `c234` caiu para 24,94%. São resultados do mesmo holdout interno, não OOF, e não substituem a referência lexical. Veja o [histórico experimental](../desenvolvimento.md) e o [relatório BERTimbau](experimento-bertimbau.md).

| Nome | Representação e classificador | Configuração de busca |
|---|---|---|
| majority | Classe majoritária aprendida em cada treino | Referência automática |
| baseline | TF-IDF de palavras + regressão logística, C=2 | Referência fixa |
| candidate | TF-IDF palavras/caracteres + SVM, versão anterior | Modelo fixo |
| lexical | TF-IDF palavras/caracteres + regressão logística | classical.json |
| word_svc | TF-IDF de palavras + LinearSVC | linear-svc.json |
| linear_svc | TF-IDF palavras/caracteres + LinearSVC | linear-svc.json |
| static | POTION multilíngue congelado + regressão logística | static.json |
| static_hybrid | TF-IDF de palavras + POTION, peso semântico 3 + regressão logística | static.json |
| semantic | MiniLM multilíngue congelado + regressão logística | semantic.json |
| semantic_mlp | Mesmo MiniLM congelado + MLP de uma camada | semantic-mlp.json |
| hybrid | TF-IDF palavras/caracteres + MiniLM + regressão logística | semantic.json |

Os modelos clássicos não atendem, por si só, ao critério de inovação do EP. As alternativas com embeddings são candidatas a uma abordagem mais elaborada; desempenho e avaliação acadêmica precisam ser verificados.

### Semântica eficiente em CPU: Model2Vec/POTION

```bash
python -m pip install -r requirements/research/static.txt
HF_HOME=.cache/huggingface PYTHONPATH=src python -m model_research.cli embed --train data/train.xlsx --backend static
HF_HUB_OFFLINE=1 PYTHONPATH=src python -m model_research.cli experiment \
  --train data/train.xlsx --config configs/research/static.json \
  --folds 3 --inner-folds 3 --threads 2 \
  --output-dir results/research/meu-experimento-semantico
```

O [POTION multilíngue](https://huggingface.co/minishlab/potion-multilingual-128M) fornece embeddings estáticos de 256 dimensões, destilados de BGE-M3. A implementação processa o texto inteiro, sem truncamento, e normaliza os vetores. A revisão dos pesos é fixada no código. O primeiro uso baixa cerca de 530 MB. A inferência ocorre localmente.

### Semântica contextual: MiniLM

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/research/semantic.txt
HF_HOME=.cache/huggingface PYTHONPATH=src python -m model_research.cli embed --train data/train.xlsx --backend transformer
HF_HUB_OFFLINE=1 PYTHONPATH=src python -m model_research.cli experiment \
  --train data/train.xlsx --config configs/research/semantic.json \
  --folds 3 --inner-folds 3 --threads 2 \
  --output-dir results/research/meu-experimento-contextual
```

O [MiniLM multilíngue](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) produz 384 dimensões. Respostas longas são divididas em janelas de tokens, com até três janelas distribuídas entre início, meio e fim; janelas intermediárias podem ser omitidas. Fazemos a média dos vetores e normalização L2.

Essa opção custa mais tempo em CPU. O cache SQLite grava embeddings por texto, revisão do encoder e estratégia de janelas; uma execução interrompida retoma os textos ainda não calculados. Nenhum dos encoders é ajustado com os rótulos do EP. Apenas essas representações congeladas podem ser compartilhadas entre folds.

Para a GPU NVIDIA deste ambiente (MX550, 2 GB), instale o wheel CUDA e use lotes pequenos:

```bash
python -m pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
HF_HOME=.cache/huggingface PYTHONPATH=src python -m model_research.cli embed \
  --train data/train.xlsx --backend transformer --device cuda --batch-size 8
```

`--device auto` usa CUDA quando acessível e CPU caso contrário. O cálculo dos embeddings usa a GPU; os classificadores e a busca em grade usam CPU. O driver precisa estar acessível ao processo: nesta sessão ele só ficou disponível fora do isolamento. O cache é reaproveitado pela busca, sem exigir GPU para os textos já codificados.

### Diagnóstico rápido de CUDA ou ROCm

O PyTorch expõe GPUs NVIDIA (CUDA) e AMD (ROCm) pela API `torch.cuda`. Para **uso de GPU**, verifique primeiro se o wheel instalado e o driver reconhecem o dispositivo:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
PYTHONPATH=src python -m clarity.hardware
```

O diagnóstico numérico testa kernels e retropropagação; ele exige GPU e não é necessário para `predict-test --device cpu`. A interface atual do NorBERTo aceita `--device auto` (GPU disponível, senão CPU), `--device cpu` ou `--device gpu` (exige NVIDIA/CUDA ou AMD/ROCm, identificado internamente). Se a GPU for detectada mas um kernel falhar, use `--device cpu` ou corrija o ambiente; não há fallback silencioso após erro numérico. A [compatibilidade ROCm](compatibilidade-rocm.md) registra apenas o ambiente histórico desta máquina, não um pré-requisito da entrega.

`configs/research/semantic-compact.json` reduz a grade híbrida a C em {0,5; 2} e peso semântico 3. Mantém a ablação semântica e os dois níveis de validação, com menor custo que a grade completa de `semantic.json`.

## Reproduzir a referência lexical

O comando abaixo reproduz **a regressão logística de referência**, não o NorBERTo escolhido. Para o protocolo e os comandos do treino NorBERTo-512 já concluído, consulte [treino final e avaliação cruzada](../norberto/avaliacao-norberto-512-e-treino-final.md). Para reproduzir a referência lexical em um destino novo:

```bash
PYTHONPATH=src python -m model_research.cli train \
  --train data/train.xlsx --config configs/research/final-logistic.json \
  --folds 3 --seed 42 --threads 2 \
  --output artifacts/final-logistic-reproducao.joblib
```

Isso salva o pipeline ajustado e metadados adjacentes. A configuração final fixa um único modelo e `C=0,25`; a CV registrada é diagnóstica, não um novo teste nem uma nova busca adaptativa. O modelo treinado deve ser carregado com as mesmas versões de dependências; arquivos joblib precisam ser de origem confiável.

A escolha lexical da época foi encerrada pela [comparação pareada](comparacao-lexical-holdout-fixo.md); depois, o projeto escolheu o [NorBERTo-512](../modelo-escolhido-norberto-512.md) como candidato de entrega. `classical.json`, `semantic.json` e `linear-svc.json` permanecem disponíveis para reproduzir pesquisas anteriores; não os use para substituir silenciosamente a configuração escolhida.

## Preencher `data/test1.xlsx` com NorBERTo-512

A execução já produziu [`artifacts/test1-norberto-512.xlsx`](../../artifacts/test1-norberto-512.xlsx), preservando `data/test1.xlsx`. Para repetir, use **novos destinos** em `--output` e `--output-dir` (o executor recusa sobrescrita):

```bash
PYTHONPATH=src python -m clarity.finetune predict-test \
  --device auto \
  --config configs/norberto-lora-development-ctx512.json \
  --artifact-dir artifacts/norberto-lora-final-ctx512-20260928 \
  --test data/test1.xlsx \
  --output artifacts/test1-norberto-512-reproducao.xlsx \
  --output-dir results/norberto/delivery/norberto-lora-test1-reproducao
```

O modo carrega somente o adapter final, verifica configuração e hashes, e **não treina nem avalia acurácia**: `test1.xlsx` não contém rótulos verdadeiros. Use `--device cpu` para forçar CPU ou `--device gpu` para exigir GPU NVIDIA/AMD reconhecida pelo PyTorch; o backend CUDA/ROCm é escolhido internamente. O checkpoint base, preso à revisão registrada, pode ser baixado automaticamente na primeira execução; não defina `HF_HUB_OFFLINE=1` se ele ainda não estiver em cache. O [relatório de entrega](../norberto/entrega-test1-norberto-512.md) registra validação do Excel e auditoria. Docker/ROCm foi usado **apenas nesta máquina** e não acompanha a entrega.

## Exportação alternativa da referência lexical

O comando clássico `PYTHONPATH=src python -m model_research.cli predict` ainda aceita apenas `.joblib`. O exemplo abaixo gera uma planilha com a **referência lexical**, não com o NorBERTo escolhido:

```bash
PYTHONPATH=src python -m model_research.cli predict \
  --artifact artifacts/final-logistic-20260928.joblib \
  --test data/test.xlsx \
  --output artifacts/test-rotulado.xlsx
```

O exportador preserva ordem, demais células, formatação e abas; o novo modo NorBERTo reutiliza a mesma rotina de gravação e conferência. O exemplo lexical acima não deve ser usado como entrega do modelo escolhido.

## Testes

```bash
python -m pytest -q
# Opcional: integração com os dois encoders já baixados:
HF_HUB_OFFLINE=1 EP1_TEST_ENCODER=1 python -m pytest -q
```

## Estrutura e entrega

- `src/clarity/`: leitura/auditoria, modelos, encoders, experimentos, exportação e CLI.
- `configs/research/`: grades explícitas, usadas tanto na avaliação quanto no treino final.
- `results/`: métricas, metadados e predições OOF e do holdout versionados.
- `artifacts/`: somente o adapter final escolhido e seus metadados são versionados; demais modelos e checkpoints permanecem locais. `.cache/` e `.venv/` são ignorados pelo Git.
- `tests/`: invariantes dos dados, ausência de vazamento, avaliação e exportação.

A planilha fornecida `data/test1.xlsx` foi preenchida com o NorBERTo; a entrega vigente está versionada em [`deliveries/test1.xlsx`](../../deliveries/test1.xlsx). A antiga cópia local `artifacts/test1-norberto-512.xlsx` não é necessária para a entrega. Ainda faltam relatório acadêmico com identificação e números USP, link do repositório e apresentação PDF. Sem rótulos verdadeiros externos, não há acurácia oficial para `test1.xlsx`.

## Referências metodológicas

- [Validação cruzada aninhada — scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [StratifiedGroupKFold — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Model2Vec e modelos POTION](https://github.com/MinishLab/model2vec).
- [Sentence Transformers: limites de comprimento](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html).
