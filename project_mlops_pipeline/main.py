import json
import os

from src.data import load_data, prepare_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict


def main():

    # 1. Carregar dados
    points = load_data("data/raw/points.csv")

    # 2. Preparar features
    features, scaler = prepare_features(points)

    # 3. Treinar modelo
    model = train_model(features, n_clusters=5)

    # 4. Avaliar
    metrics = evaluate_model(model, features)

    # 5. Predição de exemplo (ponto fixo para validação)
    result = predict(model, scaler, lat=-23.55, lng=-46.63)

    # 6. Salvar output
    output = {"metrics": metrics, "sample_prediction": result}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Pipeline executado com sucesso!")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
