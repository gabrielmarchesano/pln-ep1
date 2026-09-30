# NorBERTo-512

Esta pasta reúne o protocolo e as avaliações do modelo escolhido. A [decisão final](../modelo-escolhido-norberto-512.md) e a [cronologia do desenvolvimento](../desenvolvimento.md) ficam na raiz de `docs/`; os pilotos anteriores e modelos alternativos estão em [pesquisa histórica](../research/README.md).

O [índice dos resultados NorBERTo](../../results/norberto/README.md) destaca o adapter **vigente** `artifacts/norberto-lora-final-ctx512-20260928/final_model/` e separa sua avaliação dos pilotos e da busca posterior não promovida.

1. [Protocolo dos três folds de desenvolvimento](experimento-norberto-512-desenvolvimento.md): configuração e critérios registrados antes da execução.
2. [Avaliação cruzada e treino final](avaliacao-norberto-512-e-treino-final.md): resultados dos folds, seleção de épocas e novo treino com todo o desenvolvimento.
3. [Avaliação separada no holdout](avaliacao-norberto-holdout-final.md): métricas do adapter final salvo, sem novo treinamento.
4. [Entrega de `test1.xlsx`](entrega-test1-norberto-512.md): inferência, auditoria e planilha em [`deliveries/test1.xlsx`](../../deliveries/test1.xlsx).
5. [Screening de hiperparâmetros](screening-hiperparametros.md): nove configurações no mesmo split, ranking e TOP 3; sem CV nesta rodada, sem usar o holdout ou substituir automaticamente o adapter final.
6. [CV dos finalistas do screening](cv-finalistas-hiperparametros.md): avaliação concluída dos três melhores; `candidate-08` venceu por acurácia média, sem substituir automaticamente o modelo de entrega.
7. [Treino final exploratório de `candidate-08`](treino-final-candidate-08.md): novo adapter salvo após três épocas em todo o desenvolvimento, sem avaliação automática do holdout.
8. [Avaliação separada de `candidate-08` no holdout](avaliacao-holdout-candidate-08.md): queda acentuada em relação ao adapter anterior; configuração não promovida.

O holdout fixo foi consultado em pesquisas anteriores; não é uma estimativa historicamente independente. `test1.xlsx` não tem gabarito, portanto sua acurácia externa não foi medida.
