# Entrega de `test1.xlsx` com NorBERTo-512

O [modelo escolhido](../modelo-escolhido-norberto-512.md) preencheu a coluna `clarity` das **900 respostas** de `data/test1.xlsx`. A planilha original permaneceu intacta. O arquivo pronto para entrega está em **[`deliveries/test1.xlsx`](../../deliveries/test1.xlsx)**. A entrada, o adapter escolhido, o tokenizer e `run.json` estão versionados no Git; somente os pesos-base da revisão fixada precisam ser baixados ou estar em cache no ambiente de inferência.

## Como executar diretamente, sem Docker

A partir da raiz do repositório, use Python 3.11–3.13 (a inferência foi validada em 3.12) e instale as dependências do modelo. **Recomendamos fortemente uma GPU NVIDIA/CUDA ou AMD/ROCm para acelerar a inferência e, sobretudo, o treinamento**; CPU funciona como alternativa mais lenta. Para GPU, o wheel `torch==2.8.0` deve corresponder ao driver/backend CUDA ou ROCm. O `Dockerfile.rocm` foi apenas um recurso local de desenvolvimento e **não acompanha a entrega**.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Após clonar o repositório, o adapter treinado, `run.json`, o tokenizer e `data/test1.xlsx` já estarão nos caminhos esperados; **não é necessário repetir o treinamento**. Na primeira execução, o código pode baixar o checkpoint base **na revisão fixada** se houver internet. Não configure `HF_HUB_OFFLINE=1` sem ter esse checkpoint em cache.

```bash
python src/main.py test \
  --device auto \
  --artifact-dir artifacts/norberto-lora-final-ctx512-20260928 \
  --test data/test1.xlsx \
  --output deliveries/test1-reproducao.xlsx \
  --output-dir results/norberto/delivery/norberto-lora-test1-reproducao
```

Em PowerShell, execute `python src/main.py test ...` com os mesmos argumentos. Tanto `train` quanto `test` aceitam `--device auto|cpu|gpu`: `auto` prefere GPU e usa CPU se ela não estiver disponível; `cpu` força CPU; `gpu` exige uma GPU funcional. O código identifica internamente NVIDIA/CUDA ou AMD/ROCm (ambas expostas pela API `torch.cuda`). O terminal mostra `DEVICE test prediction` com dispositivo, backend e nome da GPU selecionados. Para garantir aceleração e evitar um treino longo em CPU, use `--device gpu`. Se uma GPU detectada falhar no teste numérico, corrija o ambiente ou escolha `--device cpu`; não há fallback silencioso após erro numérico.

**Os destinos devem ser novos.** A entrega auditada já ocupa `deliveries/test1.xlsx` e `results/norberto/delivery/norberto-lora-test1-20260929/`; o comando acima usa caminhos novos. O modo recusa sobrescrever a entrada, a saída existente, etiquetas já preenchidas ou um adapter cujos hashes/configuração não coincidam com o treino final. Ele carrega apenas `final_model`, tokeniza com a configuração aprovada (até 512 tokens, truncamento início/fim), infere, preenche a planilha e salva as probabilidades por linha. Não treina nem usa a planilha para escolher hiperparâmetros.

A CLI histórica `PYTHONPATH=src python -m model_research.cli predict` usa modelos lexicais `.joblib` e não deve ser confundida com o NorBERTo escolhido.

## Resultado e conferência

| Item | Valor |
|---|---|
| Linhas preenchidas | 900, Excel 2–901 |
| `c1` | 240 |
| `c234` | 355 |
| `c5` | 305 |
| Tempo de inferência da entrega, AMD/ROCm local | 32,56 s |
| SHA-256 da entrada intacta | `e626f030d4bf2be889fc41dd1fe9b81fb31f1f50f3ace50530d658683260449f` |
| SHA-256 da planilha de entrega | `3dd2b68a0082f635cafa28cbeea7ab799971867ee63f40d492f26ff283898fc6` |

Os [metadados da execução](../../results/norberto/delivery/norberto-lora-test1-20260929/results.json), as [probabilidades e previsões por linha](../../results/norberto/delivery/norberto-lora-test1-20260929/predictions.csv) e a [auditoria](../../results/norberto/delivery/norberto-lora-test1-20260929/audit.json) permitem conferir o arquivo. A auditoria independente comparou todas as células das abas antes/depois: **somente as 900 células de `clarity` mudaram**; os demais valores, tipos e estilos foram preservados. Conferiu também IDs de linha, classes válidas, probabilidades finitas e normalizadas, correspondência entre `argmax` e rótulo, hashes da entrada, saída e adapter. Uma inferência real de duas linhas em **CPU sem GPU exposta** terminou e produziu as mesmas classes que a execução AMD nessas linhas. NVIDIA segue a mesma interface PyTorch, mas não foi testada aqui com esse artefato. A suíte completa encerrou com **73 testes aprovados e três pulados**.

`test1.xlsx` não traz rótulos verdadeiros: a distribuição acima é de **predições**, não uma classificação por classe real. Portanto, não é possível calcular acurácia, F1 ou matriz de confusão desse arquivo sem o gabarito externo. A [avaliação anterior do holdout](avaliacao-norberto-holdout-final.md) é distinta desta entrega.
