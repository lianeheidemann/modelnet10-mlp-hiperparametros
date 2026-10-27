# Instruções para agentes

Regras deste repositório para qualquer agente de IA que edite o notebook `notebooks/mlp_modelnet10.ipynb`.

## Imagens geradas pelo notebook

- Toda célula que gera uma imagem (gráfico, matriz de confusão, etc.) deve **salvá-la na pasta `output/`** e **sobrescrevê-la** a cada execução, para que a pasta esteja sempre atualizada.
- Usar a constante `OUTPUT_DIR` (definida na seção 0.2 do notebook, `Path('../output')`), chamando `plt.savefig` **antes** de `plt.show()`:

```python
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_nome_do_grafico.png', dpi=150, bbox_inches='tight')
plt.show()
```

- Nome do arquivo: prefixo numérico em ordem de aparição no notebook, minúsculas, sem acentos, separado por `_`, extensão `.png`.
- Nunca salvar imagens fora de `output/`. Tabelas de resultados (CSV) também vão para `output/`.
- Toda imagem de `output/` deve aparecer no `README.md`, na seção da etapa correspondente, com `<img src="output/..." alt="..." width="...">`.
- **Toda imagem do `README.md` deve ter legenda**: uma linha em itálico logo abaixo, no formato `*Figura N: o que a imagem mostra.*`, com numeração sequencial. Quando o gráfico tiver elementos que exigem explicação (pontos, linhas, cores), descrevê-los na legenda.
- **Sempre que o notebook for executado e uma imagem mudar, atualizar o `README.md` na mesma tarefa**, sem esperar pedido:
  - conferir se cada imagem de `output/` está no README e remover do README as que não existem mais;
  - atualizar no texto do README os números e conclusões que dependem da imagem (contagens, métricas, descrições);
  - manter a largura (`width`) que já estiver no README para aquela imagem.
- O README aponta para os arquivos, então a versão mais recente só aparece no GitHub depois que a pasta `output/` é commitada.
- A pasta `output/` não pode entrar no `.gitignore`.

## Notebook

- Títulos: `## Etapa N: ...` e subseções `### N.k ...`, com a mesma numeração do guia `doc/step-by-step.md`.
- Comentários no código: em português, começando com maiúscula, sem ponto final, sempre na linha acima do código (nunca ao lado) e explicando o porquê.
- Funções com docstring de uma linha.
- Ao terminar uma etapa, executar o notebook inteiro e conferir que não há erros.

## Ambiente

- Python **3.11.9** (ver `.python-version`) e as versões de `requirements.txt`.
- Seed única `SEED = 42` em folds, buscas e modelos.

## Regras do experimento

- O conjunto de teste (`X_test`, `y_test`) só é usado na avaliação final (Etapa 7), uma única vez. Nunca em buscas, validação cruzada ou ajuste da transformação.
- Baseline, Random Search e TPE usam os mesmos folds (`FOLDS`), a mesma métrica (F1 macro) e o mesmo espaço de busca.
- Manter o `README.md` atualizado: tabela de status das etapas e uma seção para cada etapa concluída.
