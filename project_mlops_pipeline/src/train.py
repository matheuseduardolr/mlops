import numpy as np
from sklearn.cluster import KMeans


def train_model(features: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> KMeans:
    """
    Treina um modelo KMeans com os features fornecidos.

    Parâmetros:
        features: array numpy normalizado de shape (n, 2)
        n_clusters: número de clusters desejado
        random_state: semente para reprodutibilidade

    Retorna:
        Modelo KMeans treinado
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(features)
    return model
