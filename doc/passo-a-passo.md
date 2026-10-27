# Guia de implementação passo a passo

Este roteiro mantém o ModelNet10 do repositório e cobre os requisitos experimentais definidos para a análise. **O dataset foi aprovado antes de começar o pré-processamento.**

## 1. Validar o dataset ✅

Apresentar nome e link do ModelNet10, objetivo de classificação, número de instâncias, representação dos dados, classes e distribuição. Aguardar a validação, pois os requisitos restringem datasets excessivamente documentados.

## 2. Carregar e conferir os dados ✅

Usar os arquivos em `dataset/modelnet10_hdf5/`:

- `modelnet10_train_0.h5`: 3.991 objetos.
- `modelnet10_test_0.h5`: 908 objetos.
- Cada objeto tem 2.048 pontos com três coordenadas (X, Y, Z) e um rótulo entre 10 classes.

Conferir dimensões, rótulos, distribuição por classe, valores ausentes e coordenadas inválidas. Manter o arquivo de teste reservado: ele não participa das buscas nem da validação cruzada.

## 3. Preparar a entrada da MLP ✅

Converter cada nuvem em uma grade binária de ocupação **8 × 8 × 8** (512 atributos). Verificar a faixa e a normalização das coordenadas; definir uma transformação determinística e aplicá-la igualmente aos dados de treino e teste. Não ajustar a transformação usando o conjunto de teste.

## 4. Definir a avaliação e o baseline ✅

Usar uma validação cruzada estratificada de **3 folds** no conjunto de treino e **F1 macro** como métrica principal. Registrar também accuracy e tempo de execução.

Treinar uma MLP com configuração fixa, sem busca de hiperparâmetros, nos mesmos folds. Esse será o baseline para medir o efeito do ajuste.

## 5. Executar Random Search e TPE ✅

Usar `MLPClassifier` dentro de um `Pipeline`, com padronização ajustada dentro de cada fold. Manter seed, folds, métrica e espaço de busca iguais nos dois métodos.

Espaço de busca sugerido:

- Camadas ocultas: `(64,)`, `(128,)` ou `(64, 64)`.
- Ativação: ReLU ou tanh.
- `alpha`: de 1e-5 a 1e-2, em escala logarítmica.
- Taxa de aprendizado: de 1e-4 a 1e-2, em escala logarítmica.
- Tamanho do lote: 32, 64 ou 128.
- Manter Adam e os demais parâmetros fixos.

Executar **25 configurações no Random Search** e **25 trials no Optuna/TPE**. Registrar os melhores parâmetros, os resultados dos trials, a evolução do TPE e sua importância estimada dos parâmetros. Interpretar essa importância como indicativa, dado o orçamento curto.

## 6. Fazer as análises de sensibilidade ✅

Cada análise varia um hiperparâmetro diferente. Variar pelo menos cinco valores, mantendo os demais na melhor configuração encontrada. Avaliar cada valor nos mesmos folds, registrar F1 macro e accuracy, fazer um gráfico e explicar o efeito observado.

## 7. Treinar e avaliar o modelo final ✅

Escolher a melhor configuração entre as buscas, treinar com todo o conjunto de treino e avaliar **uma única vez** no teste reservado. Apresentar F1 macro, accuracy, relatório por classe e matriz de confusão.

## 8. Interpretar e concluir ✅

Discutir baseline versus modelos ajustados, Random Search versus TPE, tempo de busca, sensibilidade dos hiperparâmetros e sinais de overfitting ou underfitting. Incluir no notebook autoria, decisões e a nota sobre uso de IA, se aplicável.

## Checklist de controle

- [x] Dataset validado antes do pré-processamento.
- [x] Teste mantido fora das buscas e da validação cruzada.
- [x] Baseline, Random Search e TPE avaliados nos mesmos folds.
- [x] 25 configurações em cada busca e mesmo espaço de busca.
- [x] Sensibilidade individual com pelo menos cinco valores.
- [x] Teste usado somente na avaliação final.
- [x] Resultados e limitações explicados no notebook.
