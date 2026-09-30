# NorBERTo LoRA, contexto 512, três folds de desenvolvimento

## Estado em 28/09/2026

Execução concluída e auditada em **98,57 minutos**, no contêiner `pln-norberto-512-3folds-20260928`, usando AMD RX 6650 XT (8 GB) com Docker/ROCm 6.4.1. NorBERTo: **43,47% de acurácia e 43,22% de F1 macro**, contra 43,99% e 43,54% do lexical completo. A [avaliação detalhada e o treino final](avaliacao-norberto-512-e-treino-final.md) registram a escolha por inovação, o novo ajuste nas 16.093 linhas de desenvolvimento por duas épocas e a preservação do adapter. O modelo salvo foi depois [avaliado no holdout](avaliacao-norberto-holdout-final.md) em um comando separado, sem novo treinamento.

## Protocolo registrado antes dos resultados

A configuração está em [norberto-lora-development-ctx512.json](../../configs/norberto-lora-development-ctx512.json). O objetivo é avaliar NorBERTo-512 como candidato de inovação; a avaliação histórica de 256 permanece separada e não será repetida.

- Fonte: `data/train.xlsx`, textos e rótulos originais, SHA-256 `0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818`.
- Partição: `results/research/dataset-hypotheses-20260923/splits.csv`, SHA-256 `0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494`.
- Apenas as **16.093 linhas de desenvolvimento** entram na tokenização, treino, seleção de checkpoints e avaliação OOF. As **3.999 linhas de holdout** ficam excluídas desta rodada.
- Três folds externos estratificados por grupos de similaridade da partição congelada. Cada fold utiliza a primeira divisão de uma estratificação interna em cinco partes para escolher o checkpoint por acurácia.
- Mesmo protocolo de otimização do piloto 512: revisão fixa do NorBERTo, truncamento início/fim, LoRA rank 4 e alpha 8 em `Wqkv` das 22 camadas, dropout 0,05, taxa 0,001, batch 4 com acumulação 4, até três épocas, paciência 1, FP32 e SDPA.
- Os controles baseline e lexical pareados usam exatamente os exemplos vistos pelo otimizador do encoder; o controle lexical completo usa toda a parcela de treino externo. A logística lexical dos controles usa C=0,25.
- Métricas: acurácia, F1 macro, acurácia balanceada, probabilidades, matriz de confusão e comparação pareada com o controle lexical. Nenhum resultado de fold altera a configuração dos folds seguintes.

| Fold | Treino do encoder | Validação interna | Treino externo completo | Avaliação externa |
|---|---:|---:|---:|---:|
| 1 | 8.572 | 2.199 | 10.771 | 5.322 |
| 2 | 8.769 | 2.176 | 10.945 | 5.148 |
| 3 | 8.151 | 2.319 | 10.470 | 5.623 |

Os três conjuntos externos cobrem as 16.093 linhas uma única vez. As linhas Excel originais e os grupos são preservados nos relatórios. A nova configuração exige explicitamente a partição congelada e verifica os hashes antes de treinar. O holdout já participou de avaliações lexicais anteriores; sua exclusão nesta rodada não o transforma em teste historicamente independente. Os hiperparâmetros também foram orientados por experimentos anteriores, portanto a pesquisa permanece exploratória.

## Estimativa e verificações

Estimativa de **90–110 minutos**. O piloto 512 levou 38,78 minutos, dos quais 36,84 foram de treino em 10.862 exemplos. Agora cada fold treina em aproximadamente 8,5 mil exemplos, com três avaliações externas maiores. A estimativa considera três épocas nos folds; a parada antecipada pode reduzir o tempo. O limite de segurança é 75 minutos de treino **por fold**, não uma previsão da duração total. Pico histórico no piloto 512: aproximadamente 4 GB de memória reservada pelo PyTorch.

Antes do lançamento: diagnóstico numérico ROCm aprovado; 20 testes relevantes aprovados, incluindo execução simulada dos três folds com holdout excluído, mapeamento das linhas Excel e treino/serialização real de um LoRA pequeno em CPU. Havia cerca de 192 GB livres no disco.

## Artefatos e acompanhamento

Relatórios: `results/norberto/delivery/norberto-lora-development-ctx512-20260928/`. Ao terminar, `results.json` deve apresentar `status=complete`; `oof.csv` terá as predições do desenvolvimento.

Modelos: `artifacts/norberto-lora-development-ctx512-20260928/fold-{1,2,3}/best_adapter/`. O runner salva cada checkpoint escolhido logo após o treino daquele fold, incluindo adaptadores LoRA, cabeça classificadora e tokenizer. Esses modelos podem ser recarregados para inferência sem novo treino, junto dos pesos-base da revisão fixada, presentes em `.cache/huggingface/`. Os arquivos `config.json`, `run.json` e `fold-*/training.json` registram configuração, origem e treinamento. Os adaptadores **dos folds** continuam fora do Git; a exceção versionada é somente o adapter final escolhido para entrega.

```bash
docker logs --follow --tail 100 pln-norberto-512-3folds-20260928
```

Sair dos logs com Ctrl+C não interrompe o treino. O contêiner é mantido após o término para preservar seus logs. Para conferir o estado e o código de saída:

```bash
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' pln-norberto-512-3folds-20260928
```

O comando lançado foi:

```bash
docker run -d --name pln-norberto-512-3folds-20260928 \
  --device=/dev/kfd --device=/dev/dri --shm-size=2g \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e AMDGPU_TARGETS=gfx1030 \
  -e PYTHONPATH=/workspace/src -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false -e HF_HUB_OFFLINE=1 \
  -v "$PWD:/workspace" -w /workspace pln-rocm:6.4.1 \
  python3 -u -m clarity.finetune experiment \
  --train data/train.xlsx \
  --config configs/norberto-lora-development-ctx512.json \
  --holdout-splits results/research/dataset-hypotheses-20260923/splits.csv \
  --output-dir results/norberto/delivery/norberto-lora-development-ctx512-20260928 \
  --artifact-dir artifacts/norberto-lora-development-ctx512-20260928
```

Para reproduzir futuramente, mudar o nome do contêiner e ambos os destinos; o runner rejeita diretórios não vazios.
