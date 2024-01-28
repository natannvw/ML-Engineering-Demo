import os

import pandas as pd

from model_training import train_optimize


def get_dataset(data="train"):
    data_folder = "Data"

    if data == "train":
        filename = "train.csv"

    else:
        filename = "test.csv"

    path = os.path.join(data_folder, filename)
    df = pd.read_csv(path)

    return df


def ml_pipeline():
    dataset = get_dataset(data="train")

    # Define the target and features as per the original code
    y = dataset["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(dataset[features])

    best_parameters, best_score = train_optimize(X, y)

    print("Best Parameters:", best_parameters)
    print("Best Score:", best_score)
