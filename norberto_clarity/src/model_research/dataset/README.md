# Experimentos de tratamento do dataset

Este pacote contém somente executores de pesquisa. O pipeline de produção fica em `src/clarity/` e não importa `model_research.dataset`. A planilha `data/train.xlsx` e seus rótulos permanecem originais; nenhum dos tratamentos deste pacote foi promovido.

- `cleaning.py`: rodada histórica de mascaramento simples e exclusão de conflitos exatos apenas no treino.
- `dataset_hypotheses.py`: rodada de maioria, normalização ampliada, atributos de estilo, classificação ordinal, filtro por consenso OOF e combinações.
- `audit_hypotheses.py`: verificação dos splits, rótulos, métricas e do holdout reservado na segunda rodada.

Os resultados originais estão em `results/research/dataset-cleaning-svc-20260923/` e `results/research/dataset-hypotheses-20260923/`. Caminhos antigos gravados nos manifests são resolvidos pelo código de leitura, sem links simbólicos. Eles registram os hashes dos arquivos **antes** da mudança de pacote; o commit `539c741` preserva aqueles códigos exatamente como executados. A realocação altera caminhos e hashes de fonte, mas não os artefatos históricos. Novas execuções gravam também `experiment_source_sha256` e exigem um diretório de saída novo.

Com `PYTHONPATH=src` ou no contêiner ROCm do projeto, os comandos atuais são `python -m model_research.dataset.cleaning`, `python -m model_research.dataset.dataset_hypotheses` e `python -m model_research.dataset.audit_hypotheses`. Os relatórios apresentam comandos atuais de reprodução e identificam os módulos usados nas execuções originais; para reproduzir resultados, nunca reutilize os diretórios já preenchidos.

A decisão vigente é usar o dataset original, sem correção automática de rótulos, mascaramento, filtragem ou atributos experimentais. O holdout separado na segunda rodada ficou sem avaliação durante aquele experimento, mas foi consultado posteriormente e agora funciona como validação fixa reutilizada. Por ter sido reservado após análises prévias de toda a base, não deve ser tratado como evidência historicamente independente.
