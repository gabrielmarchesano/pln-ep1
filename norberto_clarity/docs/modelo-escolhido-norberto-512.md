# Modelo escolhido — NorBERTo-base, LoRA e contexto 512

## Decisão

O candidato de entrega do EP1 é **`Itau-Unibanco/NorBERTo-base` com LoRA e contexto máximo de 512 tokens**, treinado nos textos e rótulos **originais** de `data/train.xlsx`. A revisão do modelo base é `db73446f89c96044863ea05a39f680524b84bccb`. A escolha considera a contribuição metodológica de um encoder contextual pré-treinado adaptado por LoRA e o critério de inovação do trabalho. Não foi motivada por uma superioridade preditiva demonstrada: a regressão logística lexical continua como referência forte.

Esta é a **decisão vigente**, posterior aos pilotos que favoreciam 256 tokens e à seleção da [referência lexical](research/modelo-final-logistico.md). Não reinterpreta as conclusões históricas dessas rodadas. O projeto não promoveu limpeza, mascaramento, correção automática de rótulos ou novas features ao dataset de produção.

Uma [busca posterior de hiperparâmetros](norberto/cv-finalistas-hiperparametros.md) selecionou outra configuração por pequena margem na CV, mas seu [novo adapter final caiu para 35,98% de acurácia no mesmo holdout](norberto/avaliacao-holdout-candidate-08.md), contra 45,31% deste modelo. A configuração nova **não foi promovida**; o adapter e a planilha de entrega descritos abaixo continuam vigentes. O holdout foi reutilizado e não deve ser tratado como teste independente.

## Evidência e limites

| Etapa | NorBERTo-512 | Referência lexical | Interpretação |
|---|---:|---:|---|
| Três folds no desenvolvimento, 16.093 predições OOF | 43,47% acurácia; 43,22% F1 macro | 43,99%; 43,54% | Vantagem pontual lexical; diferença de acurácia −0,53 p.p., intervalo descritivo [−1,49; +0,43] p.p. |
| Holdout fixo, 3.999 respostas, após salvar o modelo final | 45,31% acurácia; 45,31% F1 macro | 45,09%; 44,78% | Nove acertos a mais para NorBERTo; diferença +0,23 p.p., intervalo descritivo pareado [−1,51; +2,08] p.p. |

As duas etapas usam conjuntos distintos; seus percentuais não devem ser subtraídos como se fossem testes pareados. Os intervalos incluem zero, portanto não comprovam vantagem nem equivalência. No holdout, a comparação usa a logística **treinada apenas no desenvolvimento**, não o artefato lexical depois reajustado nas 20.092 linhas. O recall de `c234` melhorou de 35,00% para 42,70%, enquanto o de `c5` caiu de 57,28% para 51,40%. O holdout fixo já havia sido consultado em pesquisas lexicais; é uma avaliação do **modelo salvo**, não um teste historicamente independente. Veja a [avaliação dos folds e do treino](norberto/avaliacao-norberto-512-e-treino-final.md) e a [avaliação completa no holdout](norberto/avaliacao-norberto-holdout-final.md).

## Treino e artefato

O modo `final` reinicializou o checkpoint base; **não reutilizou o adapter de nenhum fold**. Treinou nas **16.093 linhas de desenvolvimento** e excluiu as **3.999 linhas e seus grupos** de holdout. As melhores épocas internas dos três folds foram `2, 3, 2`; a mediana fixou **duas épocas completas**, sem validação interna ou early stopping no treino final. A execução levou **31,18 minutos** no Docker/ROCm.

- Configuração: [`configs/norberto-lora-development-ctx512.json`](../configs/norberto-lora-development-ctx512.json), com LoRA rank 4, alpha 8, dropout 0,05, truncamento início/fim, batch 4, acumulação 4, taxa 0,001, FP32 e SDPA.
- Partição congelada: [`results/research/dataset-hypotheses-20260923/splits.csv`](../results/research/dataset-hypotheses-20260923/splits.csv).
- Adapter e tokenizer **versionados no Git**: [`artifacts/norberto-lora-final-ctx512-20260928/final_model/`](../artifacts/norberto-lora-final-ctx512-20260928/final_model/). O SHA-256 de `adapter_model.safetensors` é `d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`; a recarga foi auditada. O checkpoint base não está no Git e será baixado na revisão fixada, salvo se já estiver em cache.
- [Metadados do treino](../results/norberto/delivery/norberto-lora-final-ctx512-20260928/results.json), [metadados da avaliação](../results/norberto/delivery/norberto-lora-final-holdout-20260928/results.json) e [predições do holdout](../results/norberto/delivery/norberto-lora-final-holdout-20260928/holdout-predictions.csv) estão versionados. A [organização de resultados](../results/norberto/README.md) distingue esse adapter vigente dos experimentos não promovidos.

O comando separado `evaluate-holdout` carregou apenas `final_model` e **não treinou** nem ajustou configurações. O treino final não avaliou o holdout automaticamente. Os modos e comandos reproduzíveis estão no [relatório técnico](norberto/avaliacao-norberto-512-e-treino-final.md).

## Entrega de `test1.xlsx`

O comando `python src/main.py test` carrega o adapter final, infere sem treinar e usa o exportador validado de Excel. A coluna `clarity` das **900 respostas** de `data/test1.xlsx` foi preenchida em [`deliveries/test1.xlsx`](../deliveries/test1.xlsx), sem alterar a entrada; veja [instalação, comando Python direto, hashes e auditoria](norberto/entrega-test1-norberto-512.md). A CLI clássica foi isolada em `PYTHONPATH=src python -m model_research.cli` e não deve ser usada para atribuir resultados ao NorBERTo. O adapter e os pesos-base da revisão fixada precisam estar acessíveis no ambiente de inferência. A inferência usa GPU NVIDIA/AMD via PyTorch quando disponível ou CPU; o Docker ROCm usado aqui não acompanha a entrega. Como `test1.xlsx` não contém rótulos verdadeiros, sua acurácia externa ainda não pode ser calculada.
