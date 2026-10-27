# Roteiro da apresentação (até 12 minutos)

Centrada no problema e nos resultados. As figuras estão em `output/` e os números no [README](../README.md).

| Tempo | Slide | Conteúdo | Figura |
| --- | --- | --- | --- |
| 1 min | 1. Problema | Classificar objetos 3D do ModelNet10 (10 classes) com uma MLP e medir o efeito do ajuste de hiperparâmetros | n/a |
| 1 min | 2. Dados | 3.991 objetos de treino e 908 de teste, 2.048 pontos × 3 coordenadas; treino desbalanceado, teste mais equilibrado | `01_objetos_por_classe.png` |
| 1,5 min | 3. Entrada da MLP | Nuvem sem ordem vira grade de ocupação 8×8×8 (512 atributos), igual para treino e teste | `03_grade_ocupacao.png` |
| 1 min | 4. Avaliação | 3 folds estratificados, F1 macro, mesmos folds em todos os experimentos, teste reservado | n/a |
| 1 min | 5. Baseline | MLP padrão: F1 macro 0,870 | n/a |
| 2 min | 6. Buscas | Random Search e TPE, 25 configurações cada, mesmo espaço; 0,887 e 0,882 | `04_evolucao_buscas.png` |
| 1 min | 7. Importância | Taxa de aprendizado, lote e camadas pesam mais; `alpha` e ativação, pouco | `05_importancia_hiperparametros.png` |
| 1 min | 8. Sensibilidade | `alpha` de 1e-5 a 1e-2: F1 macro entre 0,883 e 0,888, dentro do ruído | `06_sensibilidade_alpha.png` |
| 1,5 min | 9. Teste final | F1 macro 0,853 e accuracy 0,857; erros entre mesas, cômodas e mesas de cabeceira | `07_matriz_confusao.png` |
| 1 min | 10. Conclusões | Ganho pequeno do ajuste, sinais de overfitting, grade grossa; limitações e declaração de uso de IA | n/a |

Total: 12 minutos.

## Pontos para destacar

- O ajuste melhora o F1 macro em 0,01 a 0,02, valor próximo do desvio entre folds (0,008).
- Random Search e TPE não se distinguem com 25 trials; os tempos variam entre execuções.
- O teste (0,853) fica abaixo da validação cruzada (0,887): overfitting e escolha da melhor entre 50 configurações.
- Limitações: grade 8×8×8, orçamento curto, uma seed, sensibilidade só do `alpha`.
