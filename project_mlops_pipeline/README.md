# MLOps Pipeline Modularization

Modularização de um pipeline de clusterização geográfica de regiões de entrega, transformando um script monolítico em módulos reutilizáveis e testáveis.



## Overview

Projeto focado na reestruturação de um pipeline de Machine Learning com **Python** e **Scikit-learn**, seguindo boas práticas de **MLOps**. O código original, escrito em um único bloco, foi separado em módulos de **dados, treino, avaliação e predição**, orquestrados por um `main.py`. O modelo agrupa **300 pontos** de latitude e longitude da região de São Paulo em **5 clusters** com **K-Means** e grava as métricas em `outputs/metrics.json`.

Projeto desenvolvido como exercício de modularização da pós-graduação em MLOps, a partir de um template **Cookiecutter**.



## Tech Stack

* Python 3.12  
* Pandas, NumPy  
* Scikit-learn (StandardScaler, KMeans, silhouette score)  
* GeoPy (distância geodésica)  
* Cookiecutter, uv  



## Estrutura do Projeto

```text
.
├── data/
│   └── raw/
│       └── points.csv     # gerado pelo generate_data.py
├── outputs/
│   └── metrics.json       # gerado pelo main.py
├── src/
│   ├── data.py            # carregamento e preparação dos dados
│   ├── train.py           # treinamento do modelo
│   ├── evaluate.py        # avaliação do modelo
│   └── predict.py         # predição
├── generate_data.py       # geração dos pontos sintéticos
├── main.py                # orquestração do pipeline
└── requirements.txt
```



## Workflow

* **Dados:** leitura do CSV com colunas `lat` e `lng`, convertido em um array de shape (n, 2).  

* **Features:** normalização com `StandardScaler`. O scaler é guardado para transformar novos pontos e inverter a normalização dos centróides.  

* **Treino:** ajuste de `KMeans(n_clusters=5, random_state=42)` sobre os pontos normalizados.  

* **Avaliação:** cálculo da inércia e do silhouette score, arredondados em 4 casas.  

* **Predição:** um ponto (lat, lng) é normalizado e atribuído ao cluster mais próximo. O centróide volta à escala original e a distância até ele é calculada em km com `geopy.distance.geodesic`.  

* **Output:** métricas e predição de exemplo salvas em `outputs/metrics.json`.  



## Resultados

Predição de exemplo para o ponto (-23.55, -46.63):

```json
{
  "metrics": {
    "inertia": 189.224,
    "silhouette_score": 0.291,
    "n_clusters": 5
  },
  "sample_prediction": {
    "cluster": 4,
    "distance_km": 12.3773
  }
}
```



## Como executar

1. Instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Gerar os dados:
   ```bash
   python generate_data.py
   ```

3. Executar o pipeline:
   ```bash
   python main.py
   ```



Autor: Matheus Eduardo Laurindo do Rego
