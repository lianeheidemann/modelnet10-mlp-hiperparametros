# Referências

## Dataset: ModelNet10

- **Fonte dos arquivos:** versão HDF5 com 2.048 pontos por objeto, obtida no [Kaggle (UVigo 3D Vision Lab 04)](https://www.kaggle.com/datasets/ivnrodrguezconde/uvigo-3d-vision-lab-04).
    - **Artigo original:** Wu et al. *3D ShapeNets*. CVPR, 2015. [arXiv:1406.5670](https://arxiv.org/abs/1406.5670).

## Modelo: MLP

- **Implementação:** [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), do scikit-learn.
- **Otimizador:** Kingma e Ba. *Adam*. ICLR, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).

## Ajuste de hiperparâmetros

| Método | Referência |
| --- | --- |
| Random Search | Bergstra e Bengio. *Random Search for Hyper-Parameter Optimization*. JMLR, 2012. [jmlr.org](https://www.jmlr.org/papers/v13/bergstra12a.html) |
| TPE | Bergstra et al. *Algorithms for Hyper-Parameter Optimization*. NeurIPS, 2011 |
| Optuna (implementação do TPE) | Akiba et al. *Optuna*. KDD, 2019. [arXiv:1907.10902](https://arxiv.org/abs/1907.10902) |
