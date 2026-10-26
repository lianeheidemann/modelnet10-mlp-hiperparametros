# Planejamento do trabalho — MLP com ModelNet10

## Objetivo

Classificar objetos 3D com uma MLP e avaliar o efeito do ajuste de hiperparâmetros.

## Dataset e validação

- Manter o ModelNet10 já armazenado em `dataset/modelnet10_hdf5/`.
- Usar os arquivos oficiais de treino (3.991 objetos) e teste (908 objetos). Cada objeto contém 2.048 pontos com coordenadas X, Y e Z; o alvo tem 10 classes.
- Antes do pré-processamento, apresentar dataset, link, dimensões e distribuição das classes à professora para validação. A lauda exige essa aprovação e restringe datasets muito documentados; confirmar que o ModelNet10 é aceito antes de prosseguir.

## Preparação e representação

- Analisar distribuição das classes, valores das coordenadas e dados ausentes ou inválidos.
- Manter o teste oficial isolado durante toda a seleção de hiperparâmetros.
- Converter cada nuvem de pontos em uma grade de ocupação 3D de **8 × 8 × 8**, gerando 512 atributos por objeto. Aplicar a mesma transformação ao treino e ao teste.
- Fazer validação cruzada estratificada com **3 folds** somente no conjunto de treino. Ajustar a padronização dentro de cada fold.

## MLP e experimentos

Usar `MLPClassifier` do scikit-learn em um `Pipeline`. Comparar:

1. **Baseline sem busca:** MLP com configuração fixa padrão, avaliada nos mesmos folds.
2. **Random Search:** 25 configurações.
3. **TPE com Optuna:** 25 trials.

Usar o mesmo espaço de busca e os mesmos folds nos dois métodos: arquitetura (64), (128) ou (64, 64); ativação ReLU ou tanh; regularização `alpha` entre 1e-5 e 1e-2; taxa de aprendizado entre 1e-4 e 1e-2; lote de 32, 64 ou 128. Manter o otimizador Adam e os demais parâmetros fixos.

A métrica principal será **F1 macro**; accuracy será complementar. Registrar as melhores configurações, resultados, evolução do TPE e importância estimada dos hiperparâmetros.

## Sensibilidade e avaliação final

- Cada integrante analisa um hiperparâmetro diferente, testando pelo menos cinco valores e mantendo os demais fixos na melhor configuração.
- Apresentar os resultados em tabela e gráfico e interpretar o efeito observado.
- Treinar a configuração escolhida com todo o conjunto de treino e avaliar uma única vez no teste reservado. Relatar F1 macro, accuracy e matriz de confusão.
- Discutir baseline versus buscas, Random Search versus TPE, sensibilidade, generalização e sinais de overfitting ou underfitting.

## Entregas

- Notebook com decisões e análises em Markdown, resultados e declaração de uso de IA, se aplicável.
- Apresentação de até 12 minutos, centrada no problema, nas decisões e nos resultados — não no código.
